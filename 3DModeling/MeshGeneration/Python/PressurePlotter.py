import matplotlib.pyplot as plt
import pandas as pd

#Change these to wanted pressure volume data
file_path = r"C:\Users\alanh\Documents\CBLResearch-github\MeshGeneration\dat\lv_pressure.dat"

pressurefile = pd.read_csv(file_path, sep=r'\s+', header=0)
pressures = pressurefile.iloc[:, 1]
times = pressurefile.iloc[:, 0]

plt.plot(times, pressures, label='Pressure', color = "orange")
plt.xlabel('Time (s)')
plt.ylabel('Pressure (dyne/cm²)')
plt.title("Pressure vs. Time")
plt.grid(True)
plt.legend()

#Change this line to choose where file is saved
plt.savefig(r"C:\Users\alanh\Documents\CBLResearch-github\MeshGeneration\dat\PTimeGraph.png", dpi=300)
plt.show()