# -*- coding: utf-8 -*-
def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)


class PiecewiseSchedule(object):
    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """PiecewiseSchedule
        endpoints: [(int, int)]
            list of pairs `(time, value)`，代表time=t時的value值
        interpolation: lambda float, float, float: float
        outside_value: float
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints     = endpoints
    
    def value(self, t):
        """See Schedule.value"""
        # 假如時間t介於l_t與r_t之間，就做插值
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t <= r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)
            
        # 如果t不屬於任何pieces，返回outside value
        assert self._outside_value is not None
        return self._outside_value
    
    

class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p
        
    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)