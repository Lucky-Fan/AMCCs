#%% MFR04 Parse result
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

#%% Vehicles
# Read data
columns = ["id", "duration", "routeLength", "waitingTime", "waitingCount", "timeLoss"]
root = ET.parse('..//sumo//my_crossing//trip_output.xml')
tripinfos = root.findall('.//tripinfo')
data = [[tripinfo.get(column) for column in columns] for tripinfo in tripinfos]
data = pd.DataFrame(data=data, columns=columns)

# Filter data
good_id = {'03', '04', '05', '16', '17', '18', '08', '10', '13', '14', '21', '23', '26', '27'}
data = data.loc[data['id'].apply(lambda x: x[0:2]).isin(good_id), :]
# Transfer to numerical type
for column in columns[1:]:
    data.loc[:, column] = data[column].astype('float')

print('Total waiting time: {}'.format(data.waitingTime.sum()))
print('Total time loss: {}'.format(data.timeLoss.mean()))
print('Total waiting count: {}'.format(data.waitingCount.sum()))
print('Average waiting count: {}'.format((data.waitingCount.sum() - data.shape[0]*(35/90))/data.shape[0]))

#%% Pedestrian
loop_results2, light_results2, ped_results2 = unit_run(param,
                                                       signal_control_type='unsynchronized',
                                                       crossing_control_type='proposed',
                                                       pph=140,
                                                       seed=2,
                                                       gui=False,
                                                       trajectory='trajectory_proposed_syn.xml')

columns = ["timeLoss",]
root = ET.parse('..//sumo//my_crossing//trip_output.xml')
personinfos = root.findall('.//walk')
data = [[personinfo.get(column) for column in columns] for personinfo in personinfos]
data = pd.DataFrame(data=data, columns=columns)
for column in columns:
    data.loc[:, column] = data[column].astype('float')
data.mean()



loop_results2, light_results2, ped_results2 = unit_run(param,
                                                       signal_control_type='unsynchronized',
                                                       crossing_control_type='proposed',
                                                       pph=15,
                                                       seed=2,
                                                       gui=False,
                                                       trajectory='trajectory_proposed_syn.xml')

columns = ["timeLoss",]
root = ET.parse('..//sumo//my_crossing//trip_output.xml')
personinfos = root.findall('.//walk')
data2 = [[personinfo.get(column) for column in columns] for personinfo in personinfos]
data2 = pd.DataFrame(data=data2, columns=columns)

for column in columns:
    data2.loc[:, column] = data2[column].astype('float')


data1 = data.loc[data.timeLoss<50, :]
print('Total time loss: {}'.format(data1.timeLoss.mean()))


#%% Trajectory



