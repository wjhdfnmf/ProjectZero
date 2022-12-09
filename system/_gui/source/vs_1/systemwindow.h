#ifndef SYSTEMWINDOW_H
#define SYSTEMWINDOW_H

#include <QMainWindow>

namespace Ui {
class SystemWindow;
}

class SystemWindow : public QMainWindow
{
    Q_OBJECT

public slots:
    void update();

public:
    explicit SystemWindow(QWidget *parent = 0);
    ~SystemWindow();
    //
    bool set_exit;

private slots:
    void on_button_start_clicked();

    void on_button_pause_clicked();

    void on_button_reset_clicked();

    void on_button_export_current_clicked();

    void on_slider_speed_sliderMoved(int position);

    void on_check_trainer_toggled(bool checked);

    void on_check_export_toggled(bool checked);

    void on_spin_round_max_valueChanged(int arg1);

    void on_combo_agent_currentIndexChanged(const QString &arg1);

    void on_combo_domain_currentIndexChanged(const QString &arg1);

    void on_button_exit_clicked();

    void on_button_clear_clicked();

private:
    // variables
    int value_progress;
    int value_round_max;
    int value_round_current;
    double value_fitness_best;
    double value_fitness_avg;
    QString value_agent;
    QString value_domain;
    QString value_time_left;
    //
    QString value_status;
    //
    double set_run_speed;
    bool set_running;
    bool set_trainer_enable;
    bool set_export_enable;
    bool set_reset;
    // functions
    void setup();
    void reset();
    //
    void sample_plot();
    //
    Ui::SystemWindow *ui;
};

#endif // SYSTEMWINDOW_H
