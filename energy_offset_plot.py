import matplotlib.pyplot as plt
import numpy as np
import tfs


tw_nominal=tfs.read("twiss_nominal.tfs")
#tw_nominal=tw_nominal.set_index('NAME')
tw_off=tfs.read("twiss_corrected.tfs")
#tw_off=tw_off.set_index('NAME')

beta_nom=tw_nominal.BETX.to_numpy()
beta_off=tw_off.BETX.to_numpy()
beta_beating=(beta_off-beta_nom)/beta_nom
s=tw_nominal.S.to_numpy()

plt.title("Beta function for nominal and energy offset case")
plt.xlabel("S")
plt.ylabel("Beta funtion")
plt.plot(s, beta_nom, label="Nominal Beta")
plt.plot(s, beta_off, label="Off Beta")
plt.legend()


plt.show()
plt.title("Beta beating: energy offset")
plt.xlabel("S")
plt.ylabel("Beta beating")
plt.plot(s, beta_beating, label="Beta beating")
plt.legend()
plt.show()