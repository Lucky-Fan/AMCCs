from functions import *

# MFR03
param = dict(
    delta_t_max=35,
    v=13,
    q1=0.24, #780,
    q3=0.16, #612,
    duration=23  # The duration of a total pedestrian phase
)
E_TT = 53
# The id for those vehicles passing through the corridor
good_id = {'03', '04', '05', '16', '17', '18', '08', '10', '13', '14', '21', '23', '26', '27'}
#%% Pre-run to obtain the training data for the forecasting model
loop_results, light_results, ped_results, _ = unit_run(param, signal_control_type="fixed_unsync",
                                                       crossing_control_type="fixed",
                                                       pph=30, seed=1, trajectory='Train_traj.xml', filter_edge=False,
                                                       )
#
forecast_models = train_forecast_model(loop_results, base_path+'Train_traj.xml')
with open('model_fan.pickle', 'wb') as f:
     pickle.dump(forecast_models, f)

with open('model_fan.pickle', 'rb') as f:
    forecast_models = pickle.load(f)

loop_results, light_results, ped_results, _ = unit_run(param, signal_control_type="unsynchronized",
                                                       crossing_control_type="proposed_cv",
                                                       pph=30, seed=1, forecast_models=forecast_models, gui=True)


#%% Output related results: ATT, AVD, AVS, average pedestrian delay (APDD), average person delay (APRD)
# Containers for results
ATT_result = []
AVD_result = []
AVS_result = []
APDD_result = []
APRD_result = []
# Simulate for 20 times when pph is 50 at 3 different control types and save results
for i in range(20):
    loop_results, light_results, ped_results, _ = unit_run(param, signal_control_type="fixed_unsync",
                                                        crossing_control_type="fixed",
                                                        pph=30, seed=i)
    # loop_results, light_results, ped_results, _ = unit_run(param, signal_control_type="unsynchronized",
    #                                                     crossing_control_type="proposed",
    #                                                     pph=5, seed=i)
    # Calculate total travel time from intersection T01 to T03 (W->E)
    O_TLP18_data = pd.concat([loop_result_to_dataframe(loop_results['TLP18_0']),
                            loop_result_to_dataframe(loop_results['TLP18_1'])])
    F18_out_data = O_TLP18_data.drop(columns=['t_start'])
    I_TLP17_data = pd.concat([loop_result_to_dataframe(loop_results['-TLP17_0']),
                            loop_result_to_dataframe(loop_results['-TLP17_1'])])
    F17_in_data = I_TLP17_data.drop(columns=['t_end'])
    final1 = F18_out_data.merge(F17_in_data, left_on='vID', right_on='vID')
    travel_time_data1 = final1.t_end - final1.t_start
    # Calculate total travel time from intersection T03 to T01 (E->W)
    O_TLP17_data = pd.concat([loop_result_to_dataframe(loop_results['TLP17_0']),
                            loop_result_to_dataframe(loop_results['TLP17_1'])])
    F17_out_data = O_TLP17_data.drop(columns=['t_start'])
    I_TLP18_data = pd.concat([loop_result_to_dataframe(loop_results['-TLP18_0']),
                            loop_result_to_dataframe(loop_results['-TLP18_1'])])
    F18_in_data = I_TLP18_data.drop(columns=['t_end'])
    final2 = F17_out_data.merge(F18_in_data, left_on='vID', right_on='vID')
    travel_time_data2 = final2.t_end - final2.t_start
    # Merge two directions
    travel_time_data = travel_time_data1.append(travel_time_data2)
    # Average vehicle travel time
    ATT_result.append(travel_time_data.mean())
    # Average vehicle delay E_TT = 53
    AVD_result.append(travel_time_data.mean()-E_TT)
    # Average stop times
    columns = ["id", "waitingCount"]
    root = ET.parse('..//sumo//my_crossing//trip_output.xml')
    tripinfos = root.findall('.//tripinfo')
    stop_data = [[tripinfo.get(column) for column in columns] for tripinfo in tripinfos]
    stop_data = pd.DataFrame(data=stop_data, columns=columns)
    # Filter stop_data
    stop_data = stop_data.loc[stop_data['id'].apply(lambda x: x[0:2]).isin(good_id), :]
    # Transfer to numerical type
    stop_data.loc[:, "waitingCount"] = stop_data["waitingCount"].astype('float')
    AVS_result.append((stop_data.waitingCount.sum() - stop_data.shape[0] * (35 / 90)) / stop_data.shape[0])

    # Average pedestrian delay
    columns = ["timeLoss", ]
    root = ET.parse('..//sumo//my_crossing//trip_output.xml')
    personinfos = root.findall('.//walk')
    data = [[personinfo.get(column) for column in columns] for personinfo in personinfos]
    data = pd.DataFrame(data=data, columns=columns)
    for column in columns:
        data.loc[:, column] = data[column].astype('float')
    APDD_result.append(data.values.mean())

    # Average person delay
    APRD_result.append((1.7 * (travel_time_data.sum() - E_TT * len(travel_time_data)) + data.values.sum()) / (
            1.7 * len(travel_time_data) + data.shape[0]))

print(np.mean(ATT_result))
print(np.mean(AVD_result))
print(np.mean(AVS_result))
print(np.mean(APDD_result))
print(np.mean(APRD_result))

print(np.std(ATT_result))
print(np.std(AVD_result))
print(np.std(AVS_result))
print(np.std(APDD_result))
print(np.std(APRD_result))
#%% Vehicle travel time cumulative distribution function
pph = 30
good_id = {'03', '05', '16', '17', '08', '10', '13', '21', '26', '27'}
crossing_control_types = ["Pelican", "proposed", "fixed"]
for crossing_control_type in crossing_control_types:
    if crossing_control_type == "fixed":
        signal_control_type = 'fixed_unsync'
    else:
        signal_control_type = 'unsynchronized'
    loop_results, light_results, ped_results, _ = unit_run(param, signal_control_type=signal_control_type,
                                                        crossing_control_type=crossing_control_type,
                                                        pph=pph, seed=89)
    # Calculate total travel time from intersection T01 to T03 (W->E)
    O_TLP18_data = pd.concat([loop_result_to_dataframe(loop_results['TLP18_0']),
                            loop_result_to_dataframe(loop_results['TLP18_1'])])
    F18_out_data = O_TLP18_data.drop(columns=['t_start'])
    I_TLP17_data = pd.concat([loop_result_to_dataframe(loop_results['-TLP17_0']),
                            loop_result_to_dataframe(loop_results['-TLP17_1'])])
    F17_in_data = I_TLP17_data.drop(columns=['t_end'])
    final1 = F18_out_data.merge(F17_in_data, left_on='vID', right_on='vID')
    final1 = final1.loc[final1['vID'].apply(lambda x: x[0:2]).isin(good_id), :]
    travel_time_data1 = final1.t_end - final1.t_start

    # Calculate total travel time from intersection T03 to T01 (E->W)
    O_TLP17_data = pd.concat([loop_result_to_dataframe(loop_results['TLP17_0']),
                            loop_result_to_dataframe(loop_results['TLP17_1'])])
    F17_out_data = O_TLP17_data.drop(columns=['t_start'])
    I_TLP18_data = pd.concat([loop_result_to_dataframe(loop_results['-TLP18_0']),
                            loop_result_to_dataframe(loop_results['-TLP18_1'])])
    F18_in_data = I_TLP18_data.drop(columns=['t_end'])
    final2 = F17_out_data.merge(F18_in_data, left_on='vID', right_on='vID')
    final2 = final2.loc[final2['vID'].apply(lambda x: x[0:2]).isin(good_id), :]
    travel_time_data2 = final2.t_end - final2.t_start

    # Save vehicle travel time data
    travel_time_data1.to_csv('..//results_1//VTT_un_WE_pph{}_{}.csv'.format(pph, crossing_control_type))
    travel_time_data2.to_csv('..//results_1//VTT_un_EW_pph{}_{}.csv'.format(pph, crossing_control_type))

#%% Read data for east to west direction under uncoordination
VTT1 = pd.read_csv(f'..//results_1//VTT_un_EW_pph{pph}_fixed.csv', index_col=0).values
VTT2 = pd.read_csv(f'..//results_1//VTT_un_EW_pph{pph}_Pelican.csv', index_col=0).values
VTT3 = pd.read_csv(f'..//results_1//VTT_un_EW_pph{pph}_proposed.csv', index_col=0).values
# CDF for east to west direction under uncoordination
fig1, ax1 = plt.subplots(figsize=(4, 3))
x1 = np.sort(VTT1.ravel())
y1 = np.arange(len(VTT1))/float(len(VTT1))
x2 = np.sort(VTT2.ravel())
y2 = np.arange(len(VTT2))/float(len(VTT2))
x3 = np.sort(VTT3.ravel())
y3 = np.arange(len(VTT3))/float(len(VTT3))
plt.plot(x1,y1, label="Fixed",linestyle = "-.", color='C2')
plt.plot(x2,y2, label="Pelican", linestyle = "--", color='C1')
plt.plot(x3,y3, label="Proposed", linestyle = "-", color='C0')
ax1.margins(x=0)
plt.legend(loc='lower right')
plt.xlabel('Vehicle travel time (s)')
plt.ylabel('Cumulative frequency')
fig1.set_tight_layout(0.1)


#%% Read data for west to east direction under uncoordination
VTT4 = pd.read_csv(f'..//results_1//VTT_un_WE_pph{pph}_fixed.csv', index_col=0).values
VTT5 = pd.read_csv(f'..//results_1//VTT_un_WE_pph{pph}_Pelican.csv', index_col=0).values
VTT6 = pd.read_csv(f'..//results_1//VTT_un_WE_pph{pph}_proposed.csv', index_col=0).values
# CDF for west to east direction under coordination
fig2, ax2 = plt.subplots(figsize =(4, 3))
x4 = np.sort(VTT4.ravel())
y4 = np.arange(len(VTT4))/float(len(VTT4))
x5 = np.sort(VTT5.ravel())
y5 = np.arange(len(VTT5))/float(len(VTT5))
x6 = np.sort(VTT6.ravel())
y6 = np.arange(len(VTT6))/float(len(VTT6))
plt.plot(x4,y4, label="Fixed", linestyle = "-.", color='C2')
plt.plot(x5,y5, label="Pelican", linestyle = "--", color='C1')
plt.plot(x6,y6, label="Proposed", linestyle = "-", color='C0')
plt.legend(loc='lower right')
plt.xlabel('Vehicle travel time (s)')
plt.ylabel('Cumulative frequency')
ax2.margins(x=0)
fig2.set_tight_layout(0.1)

