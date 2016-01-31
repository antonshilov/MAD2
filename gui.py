from ui import Ui_Form
import calc


class MainWindowSlots(Ui_Form):
    def calc(self):
        function = self.function.text()
        obj_type = calc.lin
        if '**' in function:
            obj_type = calc.quad
        exp_num = int(self.exp_num.text())
        d = float(self.d.text())
        m = float(self.m.text())
        base_points = [float(self.u1_del.text()), float(self.u2_del.text()), float(self.u3_del.text())]
        intervals = [float(self.u1_int.text()), float(self.u2_int.text()), float(self.u3_int.text())]
        fish_quad = float(self.fish_quad.text())
        fish_lin = float(self.fish_lin.text())
        stud_quad = float(self.stud_quad.text())
        stud_lin = float(self.stud_lin.text())
        coch_quad = float(self.koh_quad.text())
        coch_lin = float(self.koh_lin.text())
        var_num = 2
        if self.num2.isChecked():
            var_num = 2
        res_lin = calc.build_model(obj_type, calc.lin, var_num, exp_num, base_points, intervals, function,
                                   coch_lin, fish_lin, stud_lin, m, d)
        res_quad = calc.build_model(obj_type, calc.quad, var_num, exp_num, base_points, intervals, function,
                                    coch_quad, fish_quad, stud_quad, m, d)

        self.linear_model.setText(res_lin['model_function'])
        self.linear_model_adeq.setText(str(res_lin['is_model_adeq']))
        self.linear_model_adeq_d.setText(str(res_lin['adeq_disp']))
        self.quad_model.setText(res_quad['model_function'])
        self.quad_model_adeq.setText(str(res_quad['is_model_adeq']))
        self.quad_model_adeq_d.setText(str(res_quad['adeq_disp']))
        return None
