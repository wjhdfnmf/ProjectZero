#include "systemwindow.h"
#include "ui_systemwindow.h"

/*
 *
 * add:
 * - export_best_agent
 * - time_total
 *
 */

SystemWindow::SystemWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::SystemWindow)
{
    ui->setupUi(this);
    sample_plot(); // ---
    setup();

    QObject::connect(ui->button_exit, SIGNAL(clicked()),
                     QApplication::instance(), SLOT(quit()));
}

SystemWindow::~SystemWindow()
{
    delete ui;
}

void SystemWindow::setup() {
    value_status = "online";
    ui->message_out->append("online");
    // variables
    set_exit = false;
    value_round_max = 100;
    value_agent = "NONE";
    value_domain = "NONE";
    // tab labels
    ui->tab_window->setTabText(0, "System");
    ui->tab_window->setTabText(1, "Data");
    ui->tab_window->setCurrentIndex(0);
    //
    ui->line_status->setReadOnly(true);
    ui->line_time_left->setReadOnly(true);
    //
    reset();
    update();
}

void SystemWindow::reset() {
    set_running = false;
    set_reset = true;
    value_progress = 0;
    value_round_current = 1;
    value_fitness_best = 0;
    value_fitness_avg = 0;
    value_time_left = "00:00";
    //
    ui->spin_round_max->setValue(value_round_max);
}

void SystemWindow::update() {
    ui->lcd_round_max->display(value_round_max);
    ui->lcd_round_current->display(value_round_current);
    ui->lcd_fitness_best->display(value_fitness_best);
    ui->lcd_fitness_last->display(value_fitness_avg);
    ui->progressBar->setValue(value_progress);
    ui->line_time_left->setText(value_time_left);
    ui->line_status->setText(value_status);
    // check enable/disable
    if (set_running) {
        ui->button_reset->setEnabled(true);
        ui->button_pause->setEnabled(true);
        ui->button_start->setEnabled(false);
        ui->button_export_current->setEnabled(false);
        ui->combo_agent->setEnabled(false);
        ui->combo_domain->setEnabled(false);
        ui->spin_round_max->setEnabled(false);
        ui->button_export_current->setEnabled(false);
        ui->check_trainer->setEnabled(false);
        ui->check_export->setEnabled(false);
    } else if (set_reset) {
        ui->button_reset->setEnabled(false);
        ui->button_pause->setEnabled(false);
        ui->button_start->setEnabled(true);
        ui->button_export_current->setEnabled(true);
        ui->combo_agent->setEnabled(true);
        ui->combo_domain->setEnabled(true);
        ui->spin_round_max->setEnabled(true);
        ui->button_export_current->setEnabled(false);
        ui->check_trainer->setEnabled(true);
        ui->check_export->setEnabled(true);
    } else {  // paused
        ui->button_reset->setEnabled(true);
        ui->button_pause->setEnabled(false);
        ui->button_start->setEnabled(true);
        ui->button_export_current->setEnabled(true);
        ui->combo_agent->setEnabled(false);
        ui->combo_domain->setEnabled(false);
        ui->spin_round_max->setEnabled(false);
        ui->button_export_current->setEnabled(true);
        ui->check_trainer->setEnabled(false);
        ui->check_export->setEnabled(false);
    }

    QTimer::singleShot(0, this, SLOT(update()));

    //qDebug("update");
}

// -----------------------------------------------

void SystemWindow::on_button_start_clicked()
{
    // start
    set_running = true;
    set_reset = false;
    value_status = "running";
    ui->message_out->append("start");
}

void SystemWindow::on_button_pause_clicked()
{
    // pause
    set_running = false;
    value_status = "paused";
    ui->message_out->append("pause");
}

void SystemWindow::on_button_reset_clicked()
{
    // reset
    value_status = "reset";
    ui->message_out->clear();
    ui->message_out->append("reset");
    reset();
}

void SystemWindow::on_button_export_current_clicked()
{
    // export current
}

void SystemWindow::on_slider_speed_sliderMoved(int in)
{
    // change system run speed mod
    set_run_speed = in;
}

void SystemWindow::on_check_trainer_toggled(bool in)
{
    // toggle trainer
    if (in) {
        set_trainer_enable = true;
    } else {
        set_trainer_enable = false;
    }
}

void SystemWindow::on_check_export_toggled(bool in)
{
    // toggle export data
    if (in) {
        set_export_enable = true;
    } else {
        set_export_enable = false;
    }
}

void SystemWindow::on_spin_round_max_valueChanged(int arg1)
{
    // change system round_max
    value_round_max = arg1;
}

void SystemWindow::on_combo_agent_currentIndexChanged(const QString &arg1)
{
    // agent select
    value_agent = arg1;
}

void SystemWindow::on_combo_domain_currentIndexChanged(const QString &arg1)
{
    // domain select
    value_domain = arg1;
}

void SystemWindow::on_button_exit_clicked()
{
    set_exit = true;
}

void SystemWindow::on_button_clear_clicked()
{
    ui->message_out->clear();
}

// -------------------------------------------------------
void SystemWindow::sample_plot() {
    // add two new graphs and set their look:
    ui->customPlot->addGraph();
    ui->customPlot->graph(0)->setPen(QPen(Qt::blue)); // line color blue for first graph
    ui->customPlot->graph(0)->setBrush(QBrush(QColor(0, 0, 255, 20))); // first graph will be filled with translucent blue
    ui->customPlot->addGraph();
    ui->customPlot->graph(1)->setPen(QPen(Qt::red)); // line color red for second graph
    // generate some points of data (y0 for first, y1 for second graph):
    QVector<double> x(251), y0(251), y1(251);
    for (int i=0; i<251; ++i)
    {
      x[i] = i;
      y0[i] = qExp(-i/150.0)*qCos(i/10.0); // exponentially decaying cosine
      y1[i] = qExp(-i/150.0);              // exponential envelope
    }
    // configure right and top axis to show ticks but no labels:
    // (see QCPAxisRect::setupFullAxesBox for a quicker method to do this)
    ui->customPlot->xAxis2->setVisible(true);
    ui->customPlot->xAxis2->setTickLabels(false);
    ui->customPlot->yAxis2->setVisible(true);
    ui->customPlot->yAxis2->setTickLabels(false);
    // make left and bottom axes always transfer their ranges to right and top axes:
    connect(ui->customPlot->xAxis, SIGNAL(rangeChanged(QCPRange)), ui->customPlot->xAxis2, SLOT(setRange(QCPRange)));
    connect(ui->customPlot->yAxis, SIGNAL(rangeChanged(QCPRange)), ui->customPlot->yAxis2, SLOT(setRange(QCPRange)));
    // pass data points to graphs:
    ui->customPlot->graph(0)->setData(x, y0);
    ui->customPlot->graph(1)->setData(x, y1);
    // let the ranges scale themselves so graph 0 fits perfectly in the visible area:
    ui->customPlot->graph(0)->rescaleAxes();
    // same thing for graph 1, but only enlarge ranges (in case graph 1 is smaller than graph 0):
    ui->customPlot->graph(1)->rescaleAxes(true);
    // Note: we could have also just called ui->customPlot->rescaleAxes(); instead
    // Allow user to drag axis ranges with mouse, zoom with mouse wheel and select graphs by clicking:
    ui->customPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables);
}
