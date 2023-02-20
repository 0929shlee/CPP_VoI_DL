

class AoI:
    def __init__(self, age_fn, ltt):
        self.age_fn = age_fn
        self.ltt = ltt
        self.age_fn_integral = self.__get_integral_function(age_fn)

    def __f_range(self, start, end, step):
        res = []
        n = start
        while True:
            if n >= end:
                break
            res.append(n)
            n += step
        return res

    def __get_integral_function(self, fn):
        dx = 1e-2

        def res_fn(lb, ub):
            res = 0.0
            for x in self.__f_range(lb, ub + dx, dx):
                mid_y = (fn(x - dx/2) + fn(x + dx/2)) / 2.0
                res += (mid_y * dx)
            return res

        return res_fn

    def get_voi(self, n_arr, freq):
        res = 0.0
        time_step = 1.0 / freq
        for n in n_arr:
            res += self.age_fn_integral(0.0, n)
            if n > time_step:
                res -= self.age_fn_integral(0.0, n - time_step)
        
        return res * len(n_arr) / self.ltt

    def get_pvoi(self, n_arr, freq):
        res = 0.0
        time_step = 1.0 / freq
        for n in n_arr:
            res += self.age_fn(n)
        
        return res * len(n_arr) / self.ltt

