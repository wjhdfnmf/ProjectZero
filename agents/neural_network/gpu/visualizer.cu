// visualizer.cu
//
// Booster Landing 3D visualizer (Windows desktop, OpenGL, no dependencies).
//
// Flies trained networks LIVE: each frame it runs the same physics (rocket.cuh)
// and the same network math (forward_serial in gpu_nn.cuh) as training, on the
// CPU, on the same held-out test scenarios used in the reports. The problem is
// 2D (motion in a vertical plane); the visualizer places that plane in a 3D
// scene (ocean, drone ship, booster, engine flame, wind) and adds cameras.
//
// usage:
//   visualizer.exe [weights.txt ...]       default: the three flagship models in outputs/
//   visualizer.exe --check 200             headless: re-fly test scenarios, compare with test.csv
//   visualizer.exe --shot out.ppm --cam 1 --time 20 --scenario 0 --model 2
//                                          render one frame to a PPM image and exit
//
// keys:  1-6 camera   C next camera   Space pause   N / P next / previous scenario
//        R restart    M next model    Up / Down speed    T trail    H help    Esc quit
// mouse: drag = orbit camera (camera 4), wheel = zoom
// --------------------

#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <GL/gl.h>
#include <GL/glu.h>

#include "gpu_nn.cuh"
#include "rocket.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using gnn::Topology;
using rocket::Scenario;
using rocket::State;

static const float PI = 3.14159265f;

// ------------------------------------------------------------------
// Models and scenarios (scenario generator identical to rocket.cu)
// ------------------------------------------------------------------
struct Model {
	std::string name, path;
	Topology t;
	std::vector<float> w;
	rocket::Params P;
	int hist = 0;
	int landed = 0, flown = 0;  // tally while viewing
};

static std::vector<Scenario> make_scenarios(int n, uint32_t seed, bool hard, bool real, float ship_amp = 5.0f,
                                            float heave_amp = 1.5f) {
	std::mt19937 r(seed);
	auto U = [&](float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(r); };
	std::vector<Scenario> v(n);
	for (auto& s : v) {
		s.x = U(-600, 600);   s.y = U(1500, 2500);
		s.vx = U(-40, 40);    s.vy = U(-90, -50);
		s.theta = U(-0.25f, 0.25f); s.omega = U(-0.05f, 0.05f);
		s.fuel = U(4000, 7000);
		s.wind = U(-15, 15);  s.gust_amp = U(0, 8); s.gust_w = U(0.2f, 1.0f); s.gust_phase = U(0, 6.2832f);
		s.ship_amp = 0.0f; s.ship_w = 0.0f; s.ship_phase = 0.0f;
		s.heave_amp = 0.0f; s.heave_w = 0.0f; s.heave_phase = 0.0f; s.noise_seed = 0;
		if (hard) {
			s.wind = U(-20, 20); s.gust_amp = U(0, 12);
			s.ship_amp = U(0, 1) * ship_amp; s.ship_w = 6.2832f / U(8, 14); s.ship_phase = U(0, 6.2832f);
		}
		if (real) {
			s.heave_amp = U(0, 1) * heave_amp; s.heave_w = 6.2832f / U(6, 10); s.heave_phase = U(0, 6.2832f);
			s.noise_seed = r();
		}
	}
	return v;
}

static bool load_model(const std::string& path, Model& m, std::string& err) {
	std::ifstream f(path);
	std::string header;
	if (!f || !std::getline(f, header)) { err = "cannot read " + path; return false; }
	std::stringstream hs(header);
	std::string shape, word, mode;
	hs >> shape;
	while (hs >> word) { if (word == "hist") m.hist = 1; else mode = word; }
	std::vector<int> sizes;
	{ std::stringstream ss(shape); std::string x; while (std::getline(ss, x, '-')) sizes.push_back(std::stoi(x)); }
	m.t = gnn::make_topology(sizes);
	// mode from the header, or from the input count (old files have no mode word)
	const int in = sizes[0];
	const int n_obs = m.hist ? (in - 2) / 2 : in;
	if (mode.empty()) mode = n_obs == 12 ? "real" : (n_obs == 10 ? "hard" : "easy");
	m.P.hard = (mode == "hard" || mode == "real");
	m.P.real = (mode == "real");
	if (m.P.real) { m.P.max_ignitions = 3; m.P.score = 1; m.P.noise = 1.0f; m.P.delay = 2; }
	if (rocket::n_inputs(m.P, m.hist) != in) { err = path + ": input count does not match mode " + mode; return false; }
	m.w.resize(m.t.n_params);
	for (float& v : m.w) if (!(f >> v)) { err = path + ": too few weights"; return false; }
	m.path = path;
	m.name = mode + " mode, " + gnn::describe(m.t) + (m.hist ? " +memory" : "");
	return true;
}

// ------------------------------------------------------------------
// One live flight
// ------------------------------------------------------------------
struct Frame { float x, y, theta, thr, gim, pad_x, pad_y; };

struct Flight {
	const Model* m = nullptr;
	Scenario sc{};
	State st{};
	float mem[rocket::MEM_FLOATS];
	float in[32], out[8];
	std::vector<float> ba, bb;
	std::vector<Frame> trail;
	float thr_cmd = 0, gim = 0;

	void start(const Model* model, const Scenario& s) {
		m = model; sc = s;
		st = rocket::start(sc);
		ba.assign(m->t.max_width, 0); bb.assign(m->t.max_width, 0);
		for (float& v : mem) v = 0;
		trail.clear();
		thr_cmd = 0; gim = 0;
		trail.push_back({st.x, st.y, st.theta, 0, 0, st.pad_x, st.pad_y});
	}
	bool flying() const { return st.outcome == rocket::FLYING; }
	void step() {
		if (!flying()) return;
		rocket::make_input(st, m->P, m->hist, mem, in);
		gnn::forward_serial(m->t, m->w.data(), in, out, ba.data(), bb.data());
		rocket::command(out, m->P, thr_cmd, gim);
		rocket::step(st, thr_cmd, gim, sc, m->P);
		rocket::remember_command(st, m->P, m->hist, mem, gim);
		trail.push_back({st.x, st.y, st.theta, st.thr, gim, st.pad_x, st.pad_y});
	}
	float wind() const { return sc.wind + sc.gust_amp * sinf(sc.gust_w * st.t + sc.gust_phase); }
};

// ------------------------------------------------------------------
// App state
// ------------------------------------------------------------------
static HWND g_hwnd;
static HDC g_hdc;
static int g_w = 1280, g_h = 720;
static GLuint g_font = 0;
static std::vector<Model> g_models;
static int g_model = 0;
static std::vector<Scenario> g_test[3];  // per mode: easy, hard, real (seed 424242, like rocket.cu)
static int g_scen = 0;
static Flight g_flight;
static int g_cam = 1;
static bool g_paused = false, g_trail = true, g_help = true;
static const float SPEEDS[] = {0.25f, 0.5f, 1, 2, 4, 8};
static int g_speed = 2;
static double g_acc = 0;
static double g_end_timer = 0;
static float g_orbit_yaw = 0.6f, g_orbit_pitch = 0.25f, g_orbit_dist = 160;
static bool g_drag = false;
static POINT g_last_mouse;
static float g_cine = 0;
static const char* CAM_NAMES[] = {"", "Chase", "Deck", "Booster", "Orbit (drag/wheel)", "Side view (2D)", "Cinematic"};

static int mode_index(const Model& m) { return m.P.real ? 2 : (m.P.hard ? 1 : 0); }
static const std::vector<Scenario>& tests_for(const Model& m) { return g_test[mode_index(m)]; }

static void start_flight() {
	Model& m = g_models[g_model];
	const auto& T = tests_for(m);
	g_scen = (g_scen % (int)T.size() + (int)T.size()) % (int)T.size();
	g_flight.start(&m, T[g_scen]);
	g_acc = 0; g_end_timer = 0;
}

// ------------------------------------------------------------------
// Drawing helpers
// ------------------------------------------------------------------
static GLUquadric* g_q = nullptr;

static void text(float x, float y, const char* s, float r = 1, float g = 1, float b = 1) {
	glColor3f(0, 0, 0);  // drop shadow
	glRasterPos2f(x + 1, y + 1);
	glListBase(g_font - 32);
	glCallLists((GLsizei)strlen(s), GL_UNSIGNED_BYTE, s);
	glColor3f(r, g, b);
	glRasterPos2f(x, y);
	glCallLists((GLsizei)strlen(s), GL_UNSIGNED_BYTE, s);
}

static void box(float x0, float y0, float z0, float x1, float y1, float z1) {
	glBegin(GL_QUADS);
	glNormal3f(0, 1, 0);  glVertex3f(x0, y1, z0); glVertex3f(x0, y1, z1); glVertex3f(x1, y1, z1); glVertex3f(x1, y1, z0);
	glNormal3f(0, -1, 0); glVertex3f(x0, y0, z0); glVertex3f(x1, y0, z0); glVertex3f(x1, y0, z1); glVertex3f(x0, y0, z1);
	glNormal3f(0, 0, 1);  glVertex3f(x0, y0, z1); glVertex3f(x1, y0, z1); glVertex3f(x1, y1, z1); glVertex3f(x0, y1, z1);
	glNormal3f(0, 0, -1); glVertex3f(x0, y0, z0); glVertex3f(x0, y1, z0); glVertex3f(x1, y1, z0); glVertex3f(x1, y0, z0);
	glNormal3f(1, 0, 0);  glVertex3f(x1, y0, z0); glVertex3f(x1, y1, z0); glVertex3f(x1, y1, z1); glVertex3f(x1, y0, z1);
	glNormal3f(-1, 0, 0); glVertex3f(x0, y0, z0); glVertex3f(x0, y0, z1); glVertex3f(x0, y1, z1); glVertex3f(x0, y1, z0);
	glEnd();
}

// cylinder along +Y from y=0 to y=h
static void cyl(float r0, float r1, float h, int slices = 24) {
	glPushMatrix();
	glRotatef(-90, 1, 0, 0);
	gluCylinder(g_q, r0, r1, h, slices, 1);
	glPopMatrix();
}

static void draw_sky() {
	glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0, 1, 0, 1, -1, 1);
	glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();
	glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING); glDisable(GL_FOG);
	glBegin(GL_QUADS);
	glColor3f(0.55f, 0.72f, 0.90f); glVertex2f(0, 0); glVertex2f(1, 0);
	glColor3f(0.13f, 0.32f, 0.62f); glVertex2f(1, 1); glVertex2f(0, 1);
	glEnd();
	glEnable(GL_DEPTH_TEST); glEnable(GL_FOG);
	glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW);
}

static void draw_ocean(bool land) {
	glDisable(GL_LIGHTING);
	if (land) glColor3f(0.42f, 0.47f, 0.36f); else glColor3f(0.08f, 0.26f, 0.42f);
	glBegin(GL_QUADS);
	glVertex3f(-20000, -0.5f, -20000); glVertex3f(-20000, -0.5f, 20000);
	glVertex3f(20000, -0.5f, 20000);   glVertex3f(20000, -0.5f, -20000);
	glEnd();
	// grid for motion and scale (100 m)
	glColor3f(land ? 0.36f : 0.12f, land ? 0.41f : 0.33f, land ? 0.30f : 0.50f);
	glBegin(GL_LINES);
	for (int i = -40; i <= 40; ++i) {
		glVertex3f(i * 100.0f, -0.4f, -4000); glVertex3f(i * 100.0f, -0.4f, 4000);
		glVertex3f(-4000, -0.4f, i * 100.0f); glVertex3f(4000, -0.4f, i * 100.0f);
	}
	glEnd();
	glEnable(GL_LIGHTING);
}

// drone ship (hard/real) or a landing slab (easy); deck top at deck_y
static void draw_ship(float pad_x, float deck_y, bool ship) {
	glPushMatrix();
	glTranslatef(pad_x, 0, 0);
	if (ship) {
		glColor3f(0.20f, 0.21f, 0.24f); box(-50, deck_y - 8, -26, 50, deck_y, 26);  // hull
		glColor3f(0.75f, 0.30f, 0.12f); box(-50, deck_y - 8, -26, 50, deck_y - 6, 26);  // waterline stripe
		glColor3f(0.30f, 0.31f, 0.34f); box(-50, deck_y, -26, -46, deck_y + 4, 26);    // side walls
		glColor3f(0.30f, 0.31f, 0.34f); box(46, deck_y, -26, 50, deck_y + 4, 26);
	} else {
		glColor3f(0.55f, 0.55f, 0.55f); box(-50, deck_y - 2, -50, 50, deck_y, 50);
	}
	// landing zone: 30 m wide pad (x +-15) with an X and a circle
	glDisable(GL_LIGHTING);
	glColor3f(0.95f, 0.85f, 0.15f);
	glLineWidth(3);
	glBegin(GL_LINE_LOOP);
	for (int i = 0; i < 48; ++i) glVertex3f(15 * cosf(i * 2 * PI / 48), deck_y + 0.05f, 15 * sinf(i * 2 * PI / 48));
	glEnd();
	glBegin(GL_LINES);
	glVertex3f(-9, deck_y + 0.05f, -9); glVertex3f(9, deck_y + 0.05f, 9);
	glVertex3f(-9, deck_y + 0.05f, 9);  glVertex3f(9, deck_y + 0.05f, -9);
	glEnd();
	glLineWidth(1);
	glEnable(GL_LIGHTING);
	glPopMatrix();
}

// booster: base of the engine at (x, y), tilted by theta, 40 m tall
static void draw_booster(float x, float y, float theta, float thr, float gim, bool legs_out, float t) {
	glPushMatrix();
	glTranslatef(x, y, 0);
	glRotatef(-theta * 180 / PI, 0, 0, 1);
	// engine bell
	glColor3f(0.25f, 0.25f, 0.27f);
	glPushMatrix(); glRotatef(-gim * 180 / PI, 0, 0, 1); cyl(1.4f, 0.6f, 2.5f); glPopMatrix();
	// body
	glColor3f(0.93f, 0.93f, 0.95f);
	glPushMatrix(); glTranslatef(0, 2.5f, 0); cyl(1.85f, 1.85f, 33.0f, 32); glPopMatrix();
	glColor3f(0.12f, 0.12f, 0.13f);  // interstage
	glPushMatrix(); glTranslatef(0, 35.5f, 0); cyl(1.85f, 1.85f, 4.5f, 32); glPopMatrix();
	glPushMatrix(); glTranslatef(0, 40.0f, 0); gluDisk(g_q, 0, 1.85f, 32, 1); glPopMatrix();
	// grid fins
	glColor3f(0.35f, 0.35f, 0.38f);
	box(-3.6f, 37, -0.2f, -1.8f, 39, 0.2f); box(1.8f, 37, -0.2f, 3.6f, 39, 0.2f);
	box(-0.2f, 37, -3.6f, 0.2f, 39, -1.8f); box(-0.2f, 37, 1.8f, 0.2f, 39, 3.6f);
	// landing legs: folded along the body, deployed in the last ~150 m
	glColor3f(0.15f, 0.15f, 0.16f);
	glLineWidth(4);
	glDisable(GL_LIGHTING);
	glBegin(GL_LINES);
	for (int i = 0; i < 4; ++i) {
		const float a = i * PI / 2 + PI / 4, cx = cosf(a), cz = sinf(a);
		glVertex3f(1.8f * cx, 9, 1.8f * cz);
		if (legs_out) glVertex3f(9.0f * cx, 0.0f, 9.0f * cz);
		else glVertex3f(2.0f * cx, 2.6f, 2.0f * cz);
	}
	glEnd();
	glLineWidth(1);
	// exhaust plume: points opposite the gimballed thrust, length grows with throttle
	if (thr > 0) {
		glPushMatrix();
		glRotatef(-gim * 180 / PI, 0, 0, 1);
		glRotatef(180, 1, 0, 0);
		const float flick = 1.0f + 0.08f * sinf(t * 53) + 0.05f * sinf(t * 31);
		const float len = (12 + 38 * thr) * flick;
		glDisable(GL_FOG);
		glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE);
		glColor4f(1.0f, 0.55f, 0.12f, 0.50f); cyl(1.4f, 3.5f, len);          // outer glow
		glColor4f(1.0f, 0.75f, 0.25f, 0.85f); cyl(0.9f, 1.3f, len * 0.6f);   // flame
		glColor4f(1.0f, 0.97f, 0.80f, 0.95f); cyl(0.6f, 0.2f, len * 0.25f);  // hot core
		glDepthMask(GL_TRUE); glDisable(GL_BLEND);
		glEnable(GL_FOG);
		glPopMatrix();
	}
	glEnable(GL_LIGHTING);
	glPopMatrix();
}

static void draw_shadow(float x, float y, float ground_y) {
	const float h = std::max(0.0f, y - ground_y);
	const float r = 3 + h * 0.01f, a = std::max(0.05f, 0.45f - h * 0.0002f);
	glDisable(GL_LIGHTING);
	glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA); glDepthMask(GL_FALSE);
	glColor4f(0, 0, 0, a);
	glBegin(GL_TRIANGLE_FAN);
	glVertex3f(x, ground_y + 0.1f, 0);
	for (int i = 0; i <= 32; ++i) glVertex3f(x + r * cosf(i * 2 * PI / 32), ground_y + 0.1f, r * sinf(i * 2 * PI / 32));
	glEnd();
	glDepthMask(GL_TRUE); glDisable(GL_BLEND);
	glEnable(GL_LIGHTING);
}

static void draw_trail(const std::vector<Frame>& tr) {
	glDisable(GL_LIGHTING);
	glLineWidth(2);
	glBegin(GL_LINE_STRIP);
	for (const Frame& f : tr) {
		if (f.thr > 0) glColor3f(1.0f, 0.6f, 0.2f); else glColor3f(0.85f, 0.85f, 0.9f);
		glVertex3f(f.x, f.y, 0);
	}
	glEnd();
	glLineWidth(1);
	glEnable(GL_LIGHTING);
}

static void draw_wind(float x, float y, float wind) {
	// a streak of arrows beside the booster, moving with the wind
	glDisable(GL_LIGHTING);
	glColor3f(0.85f, 0.95f, 1.0f);
	const float len = wind * 2.0f;
	glBegin(GL_LINES);
	for (int i = -1; i <= 1; ++i) {
		const float yy = y + 20 + i * 12, zz = -25;
		glVertex3f(x - len / 2, yy, zz); glVertex3f(x + len / 2, yy, zz);
		const float s = len > 0 ? 1.0f : -1.0f;
		glVertex3f(x + len / 2, yy, zz); glVertex3f(x + len / 2 - 3 * s, yy + 2, zz);
		glVertex3f(x + len / 2, yy, zz); glVertex3f(x + len / 2 - 3 * s, yy - 2, zz);
	}
	glEnd();
	glEnable(GL_LIGHTING);
}

// ------------------------------------------------------------------
// Cameras
// ------------------------------------------------------------------
static void set_camera(const Flight& fl, float dt_real) {
	const State& s = fl.st;
	const float cx = s.x, cy = s.y + 20;  // booster center
	const float px = s.pad_x, py = s.pad_y;
	float fov = 50;
	float ex, ey, ez, tx, ty, tz, ux = 0, uy = 1, uz = 0;
	switch (g_cam) {
	case 1:  // chase: off to the side and slightly behind, framed on the booster
		ex = cx + 45; ey = cy + 18; ez = 150; tx = cx; ty = cy; tz = 0; break;
	case 2: {  // deck: standing at the edge of the ship, looking up at the booster
		ex = px + 42; ey = py + 14; ez = 45; tx = cx; ty = cy - 8; tz = 0;
		fov = s.y - py > 600 ? 25.0f : 55.0f;
		break;
	}
	case 3: {  // booster cam: mounted on the side near the top, looking down past the engine
		const float sx = sinf(s.theta), cyy = cosf(s.theta);
		ex = s.x + sx * 30; ey = s.y + cyy * 30; ez = 9.0f;
		tx = s.x - sx * 60; ty = s.y - cyy * 60; tz = 30.0f;
		ux = 0; uy = 0; uz = -1;
		fov = 75;
		break;
	}
	case 4: {  // orbit: mouse controlled
		ex = cx + g_orbit_dist * cosf(g_orbit_pitch) * sinf(g_orbit_yaw);
		ey = cy + g_orbit_dist * sinf(g_orbit_pitch);
		ez = g_orbit_dist * cosf(g_orbit_pitch) * cosf(g_orbit_yaw);
		tx = cx; ty = cy; tz = 0; break;
	}
	case 5: {  // side view: far away with a narrow lens, framing booster and pad
		const float mx = 0.5f * (cx + px), my = 0.5f * (cy + py);
		const float span = std::max(250.0f, std::max(fabsf(cx - px), fabsf(cy - py)) * 1.25f);
		ez = 4000; ex = mx; ey = my; tx = mx; ty = my; tz = 0;
		fov = 2 * atanf(span / 2 / ez) * 180 / PI * 1.15f;
		break;
	}
	default: {  // cinematic: slow orbit, distance depends on altitude
		g_cine += dt_real * 0.15f;
		const float d = 120 + std::min(500.0f, (s.y - py) * 0.25f);
		ex = cx + d * sinf(g_cine); ey = cy + 25 + 0.2f * d; ez = d * cosf(g_cine);
		tx = cx; ty = cy - 10; tz = 0; break;
	}
	}
	// fog starts beyond the subject: the far side-view camera would otherwise fog everything
	const float subject = sqrtf((ex - tx) * (ex - tx) + (ey - ty) * (ey - ty) + (ez - tz) * (ez - tz));
	glFogf(GL_FOG_START, subject + 2500); glFogf(GL_FOG_END, subject + 9000);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fov, (double)g_w / g_h, 1.0, 30000.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(ex, ey, ez, tx, ty, tz, ux, uy, uz);
}

// ------------------------------------------------------------------
// HUD
// ------------------------------------------------------------------
static void hud(const Flight& fl) {
	const State& s = fl.st;
	const Model& m = *fl.m;
	glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity(); glOrtho(0, g_w, g_h, 0, -1, 1);
	glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity();
	glDisable(GL_DEPTH_TEST); glDisable(GL_LIGHTING); glDisable(GL_FOG);
	// panel backgrounds
	glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glColor4f(0, 0, 0, 0.45f);
	glBegin(GL_QUADS);
	glVertex2f(10, 10); glVertex2f(330, 10); glVertex2f(330, 300); glVertex2f(10, 300);
	glVertex2f(g_w - 430.0f, 10); glVertex2f(g_w - 10.0f, 10); glVertex2f(g_w - 10.0f, 150); glVertex2f(g_w - 430.0f, 150);
	glEnd();
	glDisable(GL_BLEND);

	char b[256];
	float y = 32;
	auto line = [&](const char* s0, float r = 1, float g = 1, float bl = 1) { text(22, y, s0, r, g, bl); y += 20; };
	snprintf(b, sizeof b, "T+ %5.1f s", s.t); line(b, 0.7f, 0.9f, 1);
	snprintf(b, sizeof b, "altitude   %7.1f m", s.y - s.pad_y); line(b);
	snprintf(b, sizeof b, "to pad     %+7.1f m", s.x - s.pad_x); line(b);
	snprintf(b, sizeof b, "descent    %7.2f m/s", -(s.vy - s.pad_vy)); line(b);
	snprintf(b, sizeof b, "sideways   %+7.2f m/s", s.vx - s.pad_vx); line(b);
	snprintf(b, sizeof b, "tilt       %+7.2f deg", s.theta * 180 / PI); line(b);
	snprintf(b, sizeof b, "throttle   %6.0f %%", s.thr * 100); line(b, s.thr > 0 ? 1.0f : 0.7f, s.thr > 0 ? 0.7f : 0.7f, s.thr > 0 ? 0.3f : 0.7f);
	snprintf(b, sizeof b, "gimbal     %+7.2f deg", fl.gim * 180 / PI); line(b);
	snprintf(b, sizeof b, "propellant %7.0f kg", s.fuel); line(b);
	if (m.P.max_ignitions > 0) { snprintf(b, sizeof b, "engine starts left  %d of %d", m.P.max_ignitions - s.ignitions, m.P.max_ignitions); line(b); }
	else { snprintf(b, sizeof b, "engine starts used  %d", s.ignitions); line(b); }
	snprintf(b, sizeof b, "wind       %+7.1f m/s (unseen)", fl.wind()); line(b, 0.8f, 0.9f, 1);
	if (m.P.hard) { snprintf(b, sizeof b, "deck       x %+5.1f m  heave %+4.2f m", s.pad_x, s.pad_y); line(b, 0.8f, 0.8f, 0.8f); }

	y = 32;
	auto rline = [&](const char* s0, float r = 1, float g = 1, float bl = 1) { text(g_w - 418.0f, y, s0, r, g, bl); y += 20; };
	snprintf(b, sizeof b, "model %d/%d: %s", g_model + 1, (int)g_models.size(), m.name.c_str()); rline(b, 1, 0.9f, 0.5f);
	snprintf(b, sizeof b, "test scenario #%d (same as test.csv)", g_scen); rline(b);
	snprintf(b, sizeof b, "camera %d: %s", g_cam, CAM_NAMES[g_cam]); rline(b);
	snprintf(b, sizeof b, "speed x%g%s", SPEEDS[g_speed], g_paused ? "  PAUSED" : ""); rline(b);
	snprintf(b, sizeof b, "landed %d of %d flights watched", m.landed, m.flown); rline(b, 0.7f, 1, 0.7f);

	if (!fl.flying()) {
		const char* names[] = {"", "LANDED", "CRASHED", "TIPPED OVER", "OUT OF BOUNDS", "TIMEOUT"};
		const bool ok = s.outcome == rocket::LANDED;
		snprintf(b, sizeof b, "%s   touchdown %.2f m/s down, %.2f m/s sideways, %.1f m from center",
		         names[s.outcome], -(s.vy - s.pad_vy), fabsf(s.vx - s.pad_vx), fabsf(s.x - s.pad_x));
		text(g_w / 2.0f - 330, g_h / 2.0f - 120, b, ok ? 0.4f : 1.0f, ok ? 1.0f : 0.4f, 0.4f);
	}
	if (g_help)
		text(14, g_h - 16.0f, "1-6 camera  C next  Space pause  N/P scenario  R restart  M model  Up/Down speed  T trail  H hide help  Esc quit",
		     0.85f, 0.85f, 0.85f);
	glEnable(GL_DEPTH_TEST); glEnable(GL_LIGHTING); glEnable(GL_FOG);
	glPopMatrix(); glMatrixMode(GL_PROJECTION); glPopMatrix(); glMatrixMode(GL_MODELVIEW);
}

// ------------------------------------------------------------------
// Frame
// ------------------------------------------------------------------
static void render(float dt_real) {
	const Flight& fl = g_flight;
	const State& s = fl.st;
	glViewport(0, 0, g_w, g_h);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	draw_sky();
	set_camera(fl, dt_real);
	const GLfloat sun[] = {0.4f, 0.8f, 0.45f, 0.0f};
	glLightfv(GL_LIGHT0, GL_POSITION, sun);
	const bool ship = fl.m->P.hard;
	draw_ocean(!ship);
	draw_ship(s.pad_x, s.pad_y, ship);
	if (g_trail) draw_trail(fl.trail);
	draw_shadow(s.x, s.y, (fabsf(s.x - s.pad_x) < 50) ? s.pad_y : 0.0f);
	draw_booster(s.x, s.y, s.theta, s.thr, fl.gim, s.y - s.pad_y < 150, s.t);
	draw_wind(s.x, s.y, fl.wind());
	hud(fl);
}

static void advance(double dt_real) {
	Flight& fl = g_flight;
	if (g_paused) return;
	if (fl.flying()) {
		g_acc += dt_real * SPEEDS[g_speed];
		const double dt = fl.m->P.dt;
		int guard = 0;
		while (g_acc >= dt && fl.flying() && guard++ < 2000) { fl.step(); g_acc -= dt; }
		if (!fl.flying()) {
			g_models[g_model].flown++;
			if (fl.st.outcome == rocket::LANDED) g_models[g_model].landed++;
		}
	} else {
		g_end_timer += dt_real;
		if (g_end_timer > 4.0) { ++g_scen; start_flight(); }  // auto-advance
	}
}

// ------------------------------------------------------------------
// Window / GL setup
// ------------------------------------------------------------------
static LRESULT CALLBACK wndproc(HWND h, UINT msg, WPARAM wp, LPARAM lp) {
	switch (msg) {
	case WM_SIZE: g_w = std::max(1, (int)LOWORD(lp)); g_h = std::max(1, (int)HIWORD(lp)); return 0;
	case WM_CLOSE: PostQuitMessage(0); return 0;
	case WM_LBUTTONDOWN: g_drag = true; GetCursorPos(&g_last_mouse); SetCapture(h); return 0;
	case WM_LBUTTONUP: g_drag = false; ReleaseCapture(); return 0;
	case WM_MOUSEMOVE:
		if (g_drag) {
			POINT p; GetCursorPos(&p);
			g_orbit_yaw -= (p.x - g_last_mouse.x) * 0.01f;
			g_orbit_pitch = std::min(1.5f, std::max(-0.3f, g_orbit_pitch + (p.y - g_last_mouse.y) * 0.01f));
			g_last_mouse = p;
			g_cam = 4;
		}
		return 0;
	case WM_MOUSEWHEEL:
		g_orbit_dist = std::min(3000.0f, std::max(40.0f, g_orbit_dist * (GET_WHEEL_DELTA_WPARAM(wp) > 0 ? 0.88f : 1.13f)));
		return 0;
	case WM_KEYDOWN:
		switch (wp) {
		case VK_ESCAPE: PostQuitMessage(0); break;
		case VK_SPACE: g_paused = !g_paused; break;
		case VK_UP: g_speed = std::min(5, g_speed + 1); break;
		case VK_DOWN: g_speed = std::max(0, g_speed - 1); break;
		case 'N': ++g_scen; start_flight(); break;
		case 'P': --g_scen; start_flight(); break;
		case 'R': start_flight(); break;
		case 'M': g_model = (g_model + 1) % (int)g_models.size(); start_flight(); break;
		case 'C': g_cam = g_cam % 6 + 1; break;
		case 'T': g_trail = !g_trail; break;
		case 'H': g_help = !g_help; break;
		default: if (wp >= '1' && wp <= '6') g_cam = (int)(wp - '0');
		}
		return 0;
	}
	return DefWindowProcA(h, msg, wp, lp);
}

static bool create_window(bool visible) {
	WNDCLASSA wc = {};
	wc.style = CS_OWNDC;
	wc.lpfnWndProc = wndproc;
	wc.hInstance = GetModuleHandleA(nullptr);
	wc.hCursor = LoadCursorA(nullptr, MAKEINTRESOURCEA(32512));  // IDC_ARROW
	wc.lpszClassName = "BoosterLandingViz";
	RegisterClassA(&wc);
	RECT r = {0, 0, g_w, g_h};
	AdjustWindowRect(&r, WS_OVERLAPPEDWINDOW, FALSE);
	g_hwnd = CreateWindowA(wc.lpszClassName, "Booster Landing - live neural network flight",
	                      WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, r.right - r.left, r.bottom - r.top,
	                      nullptr, nullptr, wc.hInstance, nullptr);
	if (!g_hwnd) return false;
	g_hdc = GetDC(g_hwnd);
	PIXELFORMATDESCRIPTOR pfd = {};
	pfd.nSize = sizeof pfd; pfd.nVersion = 1;
	pfd.dwFlags = PFD_DRAW_TO_WINDOW | PFD_SUPPORT_OPENGL | PFD_DOUBLEBUFFER;
	pfd.iPixelType = PFD_TYPE_RGBA; pfd.cColorBits = 32; pfd.cDepthBits = 24;
	SetPixelFormat(g_hdc, ChoosePixelFormat(g_hdc, &pfd), &pfd);
	wglMakeCurrent(g_hdc, wglCreateContext(g_hdc));
	// vsync if available
	typedef BOOL(WINAPI * SwapFn)(int);
	if (auto sw = (SwapFn)wglGetProcAddress("wglSwapIntervalEXT")) sw(1);
	// font for the HUD
	HFONT font = CreateFontA(-15, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, ANSI_CHARSET, OUT_TT_PRECIS,
	                        CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY, FF_DONTCARE | DEFAULT_PITCH, "Consolas");
	SelectObject(g_hdc, font);
	g_font = glGenLists(96);                    // list g_font + (c - 32) draws character c
	wglUseFontBitmapsA(g_hdc, 32, 96, g_font);
	// GL state
	g_q = gluNewQuadric();
	gluQuadricNormals(g_q, GLU_SMOOTH);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING); glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL); glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_NORMALIZE);
	const GLfloat amb[] = {0.35f, 0.35f, 0.40f, 1}, dif[] = {0.85f, 0.82f, 0.75f, 1};
	glLightfv(GL_LIGHT0, GL_AMBIENT, amb); glLightfv(GL_LIGHT0, GL_DIFFUSE, dif);
	glEnable(GL_FOG);
	const GLfloat fog[] = {0.50f, 0.66f, 0.85f, 1};
	glFogfv(GL_FOG_COLOR, fog); glFogi(GL_FOG_MODE, GL_LINEAR); glFogf(GL_FOG_START, 2500); glFogf(GL_FOG_END, 9000);
	glEnable(GL_LINE_SMOOTH);
	if (visible) ShowWindow(g_hwnd, SW_SHOW);
	return true;
}

// ------------------------------------------------------------------
// Headless check: re-fly test scenarios and compare outcomes with test.csv
// ------------------------------------------------------------------
static int check(int n) {
	int bad_total = 0;
	for (Model& m : g_models) {
		std::string csv = m.path.substr(0, m.path.find_last_of("/\\") + 1) + "test.csv";
		std::ifstream f(csv);
		if (!f) { printf("%s: no test.csv next to the weights, skipped\n", m.name.c_str()); continue; }
		std::string line; std::getline(f, line);
		std::vector<int> ref;
		while (std::getline(f, line)) { std::stringstream ss(line); std::string a, b; std::getline(ss, a, ','); std::getline(ss, b, ','); ref.push_back(std::stoi(b)); }
		const auto& T = tests_for(m);
		int same = 0, landed = 0, k = std::min(n, (int)ref.size());
		for (int i = 0; i < k; ++i) {
			Flight fl; fl.start(&m, T[i]);
			while (fl.flying()) fl.step();
			same += fl.st.outcome == ref[i];
			landed += fl.st.outcome == rocket::LANDED;
		}
		printf("%-34s first %d test scenarios: same outcome as training's test.csv on %d/%d (landed %d)\n",
		       m.name.c_str(), k, same, k, landed);
		bad_total += k - same;
	}
	return bad_total > 0 ? 2 : 0;
}

// ------------------------------------------------------------------
// main
// ------------------------------------------------------------------
int main(int argc, char** argv) {
	std::vector<std::string> paths;
	int check_n = 0, shot_cam = 1, shot_model = 0;
	float shot_time = 0;
	std::string shot;
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		if (a == "--check" && i + 1 < argc) check_n = std::stoi(argv[++i]);
		else if (a == "--shot" && i + 1 < argc) shot = argv[++i];
		else if (a == "--cam" && i + 1 < argc) shot_cam = std::stoi(argv[++i]);
		else if (a == "--time" && i + 1 < argc) shot_time = std::stof(argv[++i]);
		else if (a == "--scenario" && i + 1 < argc) g_scen = std::stoi(argv[++i]);
		else if (a == "--model" && i + 1 < argc) shot_model = std::stoi(argv[++i]);
		else paths.push_back(a);
	}
	if (paths.empty()) {
		// default: the three flagship models, relative to this program's folder
		char exe[MAX_PATH]; GetModuleFileNameA(nullptr, exe, MAX_PATH);
		std::string dir(exe); dir = dir.substr(0, dir.find_last_of("\\/") + 1);
		for (const char* p : {"outputs/easy_64x64/weights.txt", "outputs/flagship/weights.txt", "outputs/flagship_real/weights.txt"})
			paths.push_back(dir + p);
	}
	for (const auto& p : paths) {
		Model m; std::string err;
		if (load_model(p, m, err)) g_models.push_back(m);
		else fprintf(stderr, "skipping: %s\n", err.c_str());
	}
	if (g_models.empty()) { fprintf(stderr, "no models loaded\n"); return 1; }
	g_test[0] = make_scenarios(1000, 424242u, false, false);
	g_test[1] = make_scenarios(1000, 424242u, true, false);
	g_test[2] = make_scenarios(1000, 424242u, true, true);
	for (const Model& m : g_models) printf("loaded %s (%s)\n", m.name.c_str(), m.path.c_str());

	if (check_n > 0) return check(check_n);

	if (!create_window(shot.empty())) { fprintf(stderr, "cannot create window\n"); return 1; }

	if (!shot.empty()) {  // render one frame at a given flight time and save it
		g_model = std::min(shot_model, (int)g_models.size() - 1);
		g_cam = shot_cam;
		start_flight();
		while (g_flight.flying() && g_flight.st.t < shot_time) g_flight.step();
		g_cine = shot_time * 0.15f;
		render(0);
		glFinish();
		std::vector<unsigned char> px((size_t)g_w * g_h * 3);
		glPixelStorei(GL_PACK_ALIGNMENT, 1);
		glReadBuffer(GL_BACK);
		glReadPixels(0, 0, g_w, g_h, GL_RGB, GL_UNSIGNED_BYTE, px.data());
		std::ofstream o(shot, std::ios::binary);
		o << "P6\n" << g_w << " " << g_h << "\n255\n";
		for (int yy = g_h - 1; yy >= 0; --yy) o.write((const char*)&px[(size_t)yy * g_w * 3], g_w * 3);
		printf("wrote %s (%dx%d, t=%.1fs, camera %d, outcome %d)\n", shot.c_str(), g_w, g_h, g_flight.st.t, g_cam, g_flight.st.outcome);
		return 0;
	}

	start_flight();
	LARGE_INTEGER freq, prev, now;
	QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&prev);
	MSG msg;
	while (true) {
		while (PeekMessageA(&msg, nullptr, 0, 0, PM_REMOVE)) {
			if (msg.message == WM_QUIT) return 0;
			TranslateMessage(&msg); DispatchMessageA(&msg);
		}
		QueryPerformanceCounter(&now);
		const double dt = std::min(0.1, (double)(now.QuadPart - prev.QuadPart) / freq.QuadPart);
		prev = now;
		advance(dt);
		render((float)dt);
		SwapBuffers(g_hdc);
	}
}
