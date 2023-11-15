from functions import *
q1 = 0.24
q3 = 0.16

# MFR03
param = dict(
    delta_t_max=35,
    v=13,
    q1=0.24,
    q3=0.16,
    duration=23  # The duration of a total pedestrian phase
)

E_TT = 53
# %% Analyze the impact of pedestrian demand
vph_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
crossing_control_types = ["fixed", "Pelican", "proposed", "proposed_cv"]


vehicle_results = {type: pd.DataFrame(index=range(20), columns=vph_list) for type in crossing_control_types}
pedestrian_results = {type: pd.DataFrame(index=range(20), columns=vph_list) for type in crossing_control_types}
person_results = {type: pd.DataFrame(index=range(20), columns=vph_list) for type in crossing_control_types}
vehicle_results_EW = {type: pd.DataFrame(index=range(20), columns=vph_list) for type in crossing_control_types}
vehicle_results_WE = {type: pd.DataFrame(index=range(20), columns=vph_list) for type in crossing_control_types}

with open('model_fan.pickle', 'rb') as f:
    forecast_models = pickle.load(f)

# Simulate and save results
for crossing_control_type in crossing_control_types:
    for vph in vph_list:
        for i in range(20):
            param['q1'] = q1*vph
            param['q3'] = q3
            print(f'vph{vph}, crossing_control_tpye{crossing_control_type}, {i}')
            if crossing_control_type == "fixed":
                signal_control_type = 'fixed_unsync'
            else:
                signal_control_type = 'unsynchronized'
            loop_results, light_results, ped_results,_ = unit_run(param, signal_control_type=signal_control_type,
                                                                crossing_control_type=crossing_control_type,
                                                                pph=40, seed=i, vph_ratio=vph,
                                                                forecast_models=forecast_models)

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
            # Average vehicle delay
            average_vehicle_d = (travel_time_data - E_TT).values.mean()
            average_vehicle_d_WE = (travel_time_data1 - E_TT).values.mean()
            average_vehicle_d_EW = (travel_time_data2 - E_TT).values.mean()

            # Average pedestrian delay
            columns = ["timeLoss", ]
            root = ET.parse('..//sumo//my_crossing//trip_output.xml')
            personinfos = root.findall('.//walk')
            data = [[personinfo.get(column) for column in columns] for personinfo in personinfos]
            data = pd.DataFrame(data=data, columns=columns)
            for column in columns:
                data.loc[:, column] = data[column].astype('float')
            average_pedestrian_d = data.values.mean()

            # Average person delay
            average_person_d = ((1.7 * (travel_time_data.sum() - E_TT * len(travel_time_data)) + data.values.sum()) / (
                        1.7 * len(travel_time_data) + data.shape[0]))

            person_results[crossing_control_type].loc[i, vph] = average_person_d
            pedestrian_results[crossing_control_type].loc[i,vph] = average_pedestrian_d
            vehicle_results[crossing_control_type].loc[i,vph] = average_vehicle_d
            vehicle_results_EW[crossing_control_type].loc[i,vph] = average_vehicle_d_EW
            vehicle_results_WE[crossing_control_type].loc[i, vph] = average_vehicle_d_WE


for crossing_control_type in crossing_control_types:
    person_results[crossing_control_type] = person_results[crossing_control_type].values.astype(np.float64)
    pedestrian_results[crossing_control_type] = pedestrian_results[crossing_control_type].values.astype(np.float64)
    vehicle_results[crossing_control_type] = vehicle_results[crossing_control_type].values.astype(np.float64)
    vehicle_results_WE[crossing_control_type] = vehicle_results_WE[crossing_control_type].values.astype(np.float64)
    vehicle_results_EW[crossing_control_type] = vehicle_results_EW[crossing_control_type].values.astype(np.float64)
# data.timeLoss = data.timeLoss.astype(np.float64)
# data120_fixed = data.copy()
# data.timeLoss.hist(bins=20)

# with open('sensitivity_vph_unsync_40.pickle', 'wb') as f:
#     pickle.dump(person_results, f)
#     pickle.dump(pedestrian_results, f)
#     pickle.dump(vehicle_results, f)
#     pickle.dump(vehicle_results_WE, f)
#     pickle.dump(vehicle_results_EW, f)
#%%
with open('sensitivity_vph_unsync_40.pickle', 'rb') as f:
    person_results = pickle.load(f)
    pedestrian_results = pickle.load(f)
    vehicle_results = pickle.load(f)
    vehicle_results_WE = pickle.load(f)
    vehicle_results_EW = pickle.load(f)

#%%
# pedestrian_results3 = {}
# person_results3 = {}
# vehicle_results3 = {}
# for crossing_control_type in crossing_control_types:
#     person_results3[crossing_control_type] = pd.concat([person_results2[crossing_control_type],
#                                                         person_results[crossing_control_type]], axis=1).values.astype(np.float64)
#     pedestrian_results3[crossing_control_type] = pd.concat([pedestrian_results2[crossing_control_type],
#                                                         pedestrian_results[crossing_control_type]], axis=1).values.astype(np.float64)
#     vehicle_results3[crossing_control_type] = pd.concat([vehicle_results2[crossing_control_type],
#                                                         vehicle_results[crossing_control_type]], axis=1).values.astype(np.float64)


#%%
vph_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
fig, ax = plt.subplots(figsize=(4,3))
ax.errorbar(x = vph_list,
            y=np.nanmean(person_results['fixed'], axis=0),
            yerr=np.nanstd(person_results['fixed'], axis=0),
            label='Fixed', linestyle = '--', color = 'C3', marker='.', capsize=3)
ax.errorbar(x = vph_list,
            y=np.nanmean(person_results['Pelican'], axis=0),
            yerr=np.nanstd(person_results['Pelican'], axis=0),
            label='Pelican', linestyle = '-.', color = 'darkorange', marker='.', capsize=3)
ax.errorbar(x = vph_list,
            y=np.nanmean(person_results['proposed'], axis=0),
            yerr=np.nanstd(person_results['proposed'], axis=0),
            label='AMCC-band', linestyle = '-', color = 'C2', marker='.', capsize=3)
ax.errorbar(x = vph_list,
            y=np.nanmean(person_results['proposed_cv'], axis=0),
            yerr=np.nanstd(person_results['proposed_cv'], axis=0),
            label='AMCC-vehicle', linestyle = '-', color = 'dodgerblue', marker='.', capsize=3)

plt.legend(loc='upper left')
ax.set_xlabel('Vehicle demand ratio')
ax.set_ylabel('Average person delay (s)')
fig.set_tight_layout(0.1)


#%%
fig, ax = plt.subplots(figsize=(4,3))
ax.errorbar(x = vph_list,
            y=np.nanmean(pedestrian_results['fixed'], axis=0),
            yerr=np.nanstd(pedestrian_results['fixed'], axis=0),
            label='Fixed', linestyle = '--', color = 'C3',marker='.', capsize=3)
ax.errorbar(x = vph_list,
            y=np.nanmean(pedestrian_results['Pelican'], axis=0),
            yerr=np.nanstd(pedestrian_results['Pelican'], axis=0),
            label='Pelican', linestyle = '-.', color = 'darkorange', marker='.', capsize=3)
ax.errorbar(x = vph_list,
            y=np.nanmean(pedestrian_results['proposed'], axis=0),
            yerr=np.nanstd(pedestrian_results['proposed'], axis=0),
            label='AMCC-band', linestyle = '-', color = 'C2', marker='.', capsize=3)
ax.errorbar(x = vph_list,
            y=np.nanmean(pedestrian_results['proposed_cv'], axis=0),
            yerr=np.nanstd(pedestrian_results['proposed_cv'], axis=0),
            label='AMCC-vehicle', linestyle = '-', color = 'dodgerblue', marker='.', capsize=3)

plt.legend(fontsize = 'xx-small', bbox_to_anchor = (0.7, 0.41))
ax.set_xlabel('Vehicle demand ratio')
ax.set_ylabel('Average pedestrian delay (s)')
fig.set_tight_layout(0.1)

#%%
fig, ax = plt.subplots(figsize=(4,3))
ax.errorbar(x = vph_list,
            y=np.nanmean(vehicle_results['fixed'], axis=0),
            yerr=np.nanstd(person_results['fixed'], axis=0),
            label='Fixed',linestyle = '--', color = 'C3', marker='.', capsize=3)
ax.errorbar(x = vph_list,
            y=np.nanmean(vehicle_results['Pelican'], axis=0),
            yerr=np.nanstd(vehicle_results['Pelican'], axis=0),
            label='Pelican', linestyle = '-.', color = 'darkorange', marker='.', capsize=3)
ax.errorbar(x = vph_list,
            y=np.nanmean(vehicle_results['proposed'], axis=0),
            yerr=np.nanstd(vehicle_results['proposed'], axis=0),
            label='AMCC-band', linestyle = '-', color = 'C2', marker='.', capsize=3)
ax.errorbar(x = vph_list,
            y=np.nanmean(vehicle_results['proposed_cv'], axis=0),
            yerr=np.nanstd(vehicle_results['proposed_cv'], axis=0),
            label='AMCC-vehicle', linestyle = '-', color = 'dodgerblue', marker='.', capsize=3)

plt.legend(loc='upper left')
ax.set_xlabel('Vehicle demand ratio')
ax.set_ylabel('Average vehicle delay (s)')
fig.set_tight_layout(0.1)

#%% Analyze the impact of vehicle demand
hp = 0.1
ms = 8
y_fixed = np.nanmean(vehicle_results_EW['fixed'], axis=0)[0:8]
x_fixed = np.nanmean(vehicle_results_WE['fixed'], axis=0)[0:8]
y_Pelican = np.nanmean(vehicle_results_EW['Pelican'], axis=0)[0:8]
x_Pelican = np.nanmean(vehicle_results_WE['Pelican'], axis=0)[0:8]
y_proposed = np.nanmean(vehicle_results_EW['proposed'], axis=0)[0:8]
x_proposed = np.nanmean(vehicle_results_WE['proposed'], axis=0)[0:8]
y_proposed_cv = np.nanmean(vehicle_results_EW['proposed_cv'], axis=0)[0:8]
x_proposed_cv = np.nanmean(vehicle_results_WE['proposed_cv'], axis=0)[0:8]

fig, ax = plt.subplots(figsize=(6,4.5))

ax.errorbar(x = x_fixed,
            y= y_fixed,
            label='Fixed',linestyle = '-.', color = 'C3', marker='.', capsize=3, ms=ms)

ax.errorbar(x = x_Pelican,
            y= y_Pelican,
            label='Pelican', linestyle = '--', color = 'darkorange', marker='.', capsize=3, ms=ms)
ax.errorbar(x = x_proposed,
            y = y_proposed,
            label='AMCC-band', linestyle = '-', color = 'C2', marker='.', capsize=3, ms=ms)
ax.errorbar(x = x_proposed_cv,
            y = y_proposed_cv,
            label='AMCC-vehicle', linestyle = '-', color = 'dodgerblue', marker='.', capsize=3, ms=ms)


for i in range(x_fixed.shape[0]):
    ax.text(x_fixed[i]+hp, y_fixed[i], vph_list[i], ha='left', va='center', color='C3')
    ax.text(x_Pelican[i]-hp, y_Pelican[i]+hp, vph_list[i], ha='right', va='bottom', color='darkorange')
    ax.text(x_proposed[i]+hp, y_proposed[i]-2.2*hp, vph_list[i], ha='center', va='top', color='C2')
    ax.text(x_proposed_cv[i]-1.5*hp, y_proposed_cv[i], vph_list[i], ha='right', va='center', color='dodgerblue')


plt.legend(loc='lower right')
ax.set_xlabel('Average vehicle delay EW (s)')
ax.set_ylabel('Average vehicle delay WE (s)')
ax.axis('equal')
fig.set_tight_layout(0.1)


