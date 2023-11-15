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
# %% Analyze the impact of pedestrian demand
# pph_list = np.arange(5, 160, 15)
pph_list = [170, 185, 200]

crossing_control_types = ["fixed", "Pelican", "proposed", "proposed_cv" ]


vehicle_results = {type: pd.DataFrame(index=range(20), columns=pph_list) for type in crossing_control_types}
pedestrian_results = {type: pd.DataFrame(index=range(20), columns=pph_list) for type in crossing_control_types}
person_results = {type: pd.DataFrame(index=range(20), columns=pph_list) for type in crossing_control_types}

with open('model_fan.pickle', 'rb') as f:
    forecast_models = pickle.load(f)

ped_ped = []
# Simulate and save results
for crossing_control_type in crossing_control_types:
    for pph in pph_list:
        for i in range(20):
            print(f'pph{pph}, crossing_control_tpye{crossing_control_type}, {i}')
            if crossing_control_type == "fixed":
                signal_control_type = 'fixed_unsync'
            else:
                signal_control_type = 'unsynchronized'
            loop_results, light_results, ped_results, _ = unit_run(param, signal_control_type=signal_control_type,
                                                                crossing_control_type=crossing_control_type,
                                                                pph=pph, seed=i, forecast_models=forecast_models)

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

            # Average pedestrian delay
            columns = ["timeLoss", ]
            root = ET.parse('..//sumo//my_crossing//trip_output.xml')
            personinfos = root.findall('.//walk')
            data = [[personinfo.get(column) for column in columns] for personinfo in personinfos]
            data = pd.DataFrame(data=data, columns=columns)
            for column in columns:
                data.loc[:, column] = data[column].astype('float')
            average_pedestrian_d = data.values.mean()
            ped_ped.append(data.copy())

            # Average person delay
            average_person_d = ((1.7 * (travel_time_data.sum() - E_TT * len(travel_time_data)) + data.values.sum()) / (
                        1.7 * len(travel_time_data) + data.shape[0]))

            person_results[crossing_control_type].loc[i, pph] = average_person_d
            pedestrian_results[crossing_control_type].loc[i,pph] = average_pedestrian_d
            vehicle_results[crossing_control_type].loc[i,pph] = average_vehicle_d

# data = pd.concat(ped_ped)
# data.timeLoss = data.timeLoss.astype(np.float64)
# data120_fixed = data.copy()
# data.timeLoss.hist(bins=20)

for crossing_control_type in crossing_control_types:
    person_results[crossing_control_type] = person_results[crossing_control_type].values.astype(np.float64)
    pedestrian_results[crossing_control_type] = pedestrian_results[crossing_control_type].values.astype(np.float64)
    vehicle_results[crossing_control_type] = vehicle_results[crossing_control_type].values.astype(np.float64)

# with open('sensitivity_pph_unsync.pickle', 'wb') as f:
#     pickle.dump(person_results, f)
#     pickle.dump(pedestrian_results, f)
#     pickle.dump(vehicle_results, f)

# with open('sensitivity_pph_unsync_extend.pickle', 'wb') as f:
#     pickle.dump(person_results, f)
#     pickle.dump(pedestrian_results, f)
#     pickle.dump(vehicle_results, f)

#%%
# with open('sensitivity_pph_unsync.pickle', 'rb') as f:
#     person_results = pickle.load(f)
#     pedestrian_results = pickle.load(f)
#     vehicle_results = pickle.load(f)
with open('sensitivity_pph_unsync_extend.pickle', 'rb') as f:
    person_results = pickle.load(f)
    pedestrian_results = pickle.load(f)
    vehicle_results = pickle.load(f)

with open('sensitivity_pph_unsync.pickle', 'rb') as f:
    person_results0 = pickle.load(f)
    pedestrian_results0 = pickle.load(f)
    vehicle_results0 = pickle.load(f)

for crossing_control_type in crossing_control_types:
    person_results[crossing_control_type] = np.concatenate([person_results0[crossing_control_type], person_results[crossing_control_type]], axis=1)
    pedestrian_results[crossing_control_type] = np.concatenate([pedestrian_results0[crossing_control_type], pedestrian_results[crossing_control_type]], axis=1)
    vehicle_results[crossing_control_type] = np.concatenate([vehicle_results0[crossing_control_type], vehicle_results[crossing_control_type]], axis=1)


#%%
# pedestrian_results3 = {}
# person_results3 = {}
# vehicle_results3 = {}
# for crossing_control_type in crossing_control_types:
#     person_results[crossing_control_type] = pd.concat([person_results[crossing_control_type],
#                                                         person_results[crossing_control_type]], axis=1).values.astype(np.float64)
#     pedestrian_results[crossing_control_type] = pd.concat([pedestrian_results2[crossing_control_type],
#                                                         pedestrian_results[crossing_control_type]], axis=1).values.astype(np.float64)
#     vehicle_results[crossing_control_type] = pd.concat([vehicle_results2[crossing_control_type],
#                                                         vehicle_results[crossing_control_type]], axis=1).values.astype(np.float64)


#%%

pph_list = np.arange(5, 210, 15)
fig, ax = plt.subplots(figsize=(4,3))
ax.errorbar(x = pph_list,
            y=np.nanmean(person_results['fixed'], axis=0),
            yerr=np.nanstd(person_results['fixed'], axis=0),
            label='Fixed', linestyle = '--', color = 'C3', marker='.', capsize=3)
ax.errorbar(x = pph_list,
            y=np.nanmean(person_results['Pelican'], axis=0),
            yerr=np.nanstd(person_results['Pelican'], axis=0),
            label='Pelican', linestyle = '-.', color = 'darkorange', marker='.', capsize=3)
ax.errorbar(x = pph_list,
            y=np.nanmean(person_results['proposed'], axis=0),
            yerr=np.nanstd(person_results['proposed'], axis=0),
            label='AMCC-band', linestyle = '-', color = 'C2', marker='.', capsize=3)
ax.errorbar(x = pph_list,
            y=np.nanmean(person_results['proposed_cv'], axis=0),
            yerr=np.nanstd(person_results['proposed_cv'], axis=0),
            label='AMCC-vehicle', linestyle = '-', color = 'dodgerblue', marker='.', capsize=3)

plt.legend(loc='lower right')
ax.set_xlabel('Pedestrian demand (pph)')
ax.set_ylabel('Average person delay (s)')
fig.set_tight_layout(0.1)


#%%
fig, ax = plt.subplots(figsize=(4,3))
ax.errorbar(x = pph_list,
            y=np.nanmean(pedestrian_results['fixed'], axis=0),
            yerr=np.nanstd(pedestrian_results['fixed'], axis=0),
            label='Fixed', linestyle = '--', color = 'C3',marker='.', capsize=3)
ax.errorbar(x = pph_list,
            y=np.nanmean(pedestrian_results['Pelican'], axis=0),
            yerr=np.nanstd(pedestrian_results['Pelican'], axis=0),
            label='Pelican', linestyle = '-.', color = 'darkorange', marker='.', capsize=3)
ax.errorbar(x = pph_list,
            y=np.nanmean(pedestrian_results['proposed'], axis=0),
            yerr=np.nanstd(pedestrian_results['proposed'], axis=0),
            label='AMCC-band', linestyle = '-', color = 'C2', marker='.', capsize=3)
ax.errorbar(x = pph_list,
            y=np.nanmean(pedestrian_results['proposed_cv'], axis=0),
            yerr=np.nanstd(pedestrian_results['proposed_cv'], axis=0),
            label='AMCC-vehicle', linestyle = '-', color = 'dodgerblue', marker='.', capsize=3)

# plt.legend(bbox_to_anchor=(0.69, 0.1))
lower, upper = ax.get_ylim()
ax.set_ybound(lower, 45)
plt.legend(loc = 'upper right')
ax.set_xlabel('Pedestrian demand (pph)')
ax.set_ylabel('Average pedestrian delay (s)')
fig.set_tight_layout(0.1)

#%%
fig, ax = plt.subplots(figsize=(4,3))
ax.errorbar(x = pph_list,
            y=np.nanmean(vehicle_results['fixed'], axis=0),
            yerr=np.nanstd(person_results['fixed'], axis=0),
            label='Fixed',linestyle = '--', color = 'C3', marker='.', capsize=3)
ax.errorbar(x = pph_list,
            y=np.nanmean(vehicle_results['Pelican'], axis=0),
            yerr=np.nanstd(vehicle_results['Pelican'], axis=0),
            label='Pelican', linestyle = '-.', color = 'darkorange', marker='.', capsize=3)
ax.errorbar(x = pph_list,
            y=np.nanmean(vehicle_results['proposed'], axis=0),
            yerr=np.nanstd(vehicle_results['proposed'], axis=0),
            label='AMCC-band', linestyle = '-', color = 'C2', marker='.', capsize=3)
ax.errorbar(x = pph_list,
            y=np.nanmean(vehicle_results['proposed_cv'], axis=0),
            yerr=np.nanstd(vehicle_results['proposed_cv'], axis=0),
            label='AMCC-vehicle', linestyle = '-', color = 'dodgerblue', marker='.', capsize=3)

plt.legend(loc='lower right')
ax.set_xlabel('Pedestrian demand (pph)')
ax.set_ylabel('Average vehicle delay (s)')
fig.set_tight_layout(0.1)


