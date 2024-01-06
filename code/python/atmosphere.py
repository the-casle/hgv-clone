import math
#from pyatmos import coesa76
def compute_g(h):
    R = 6.371e6  # radius of the earth in meters
    g_e = 9.80665  # sea level gravity acceleration

    return g_e * (R / (R + h))**2

def compute_atmosphere(h):
    #atm_res=coesa76(h/1e3)  # h should be in km
     rho_0 = 1.22500
     Beta = 0.1378  # 1/km
     R = 287.00
     T = 270.0  # Let's assume this is constant for simplicity

     rho = rho_0 * math.exp(-Beta * h / 1e3)
     P_inf = rho * (R * T)
    #return atm_res.rho, atm_res.P
     return rho, P_inf