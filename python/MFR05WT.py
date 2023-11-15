from functions import *



param = dict(
    delta_t_max=35,
    v=13,
    q1=0.24, #780,
    q3=0.16, #612,
    duration=23  # The duration of a total pedestrian phase
)
E_TT = 53

# %%

t_max_list = [17, 20, 25, 30, 35, 40, 45, 50, 55, 60]
crossing_control_types = ["proposed", "proposed_cv"]


vehicle_results = {type: pd.DataFrame(index=range(20), columns=t_max_list) for type in crossing_control_types}
pedestrian_results = {type: pd.DataFrame(index=range(20), columns=t_max_list) for type in crossing_control_types}



with open('model_fan.pickle', 'rb') as f:
    forecast_models = pickle.load(f)

for t_max in t_max_list:
    param['delta_t_max'] = t_max
    for crossing_control_type in crossing_control_types:
        for i in range(10):
            if crossing_control_type == "fixed":
                signal_control_type = 'fixed_unsync'
            else:
                signal_control_type = 'unsynchronized'
            loop_results, light_results, ped_results,_ = unit_run(param, signal_control_type=signal_control_type,
                                                                crossing_control_type=crossing_control_type,
                                                                pph=40, seed=i, forecast_models=forecast_models)

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



            pedestrian_results[crossing_control_type].loc[i,t_max] = average_pedestrian_d
            vehicle_results[crossing_control_type].loc[i,t_max] = average_vehicle_d


for crossing_control_type in crossing_control_types:
    pedestrian_results[crossing_control_type] = pedestrian_results[crossing_control_type].values.astype(np.float64)
    vehicle_results[crossing_control_type] = vehicle_results[crossing_control_type].values.astype(np.float64)

with open('sensitivity_t_max_unsync_40.pickle', 'wb') as f:
     pickle.dump(pedestrian_results, f)
     pickle.dump(vehicle_results, f)



#%%
with open('sensitivity_t_max_unsync_40.pickle', 'rb') as f:
    pedestrian_results = pickle.load(f)
    vehicle_results = pickle.load(f)

hp = 0.1
ms = 8

y_proposed = np.nanmean(vehicle_results['proposed'], axis=0)[0:8]
x_proposed = np.nanmean(pedestrian_results['proposed'], axis=0)[0:8]
y_proposed_cv = np.nanmean(vehicle_results['proposed_cv'], axis=0)[0:8]
x_proposed_cv = np.nanmean(pedestrian_results['proposed_cv'], axis=0)[0:8]

fig, ax = plt.subplots(figsize=(6,4.5))


ax.errorbar(x = x_proposed,
            y = y_proposed,
            label='AMCC-band', linestyle = '-', color = 'C2', marker='.', capsize=3, ms=ms)
ax.errorbar(x = x_proposed_cv,
            y = y_proposed_cv,
            label='AMCC-vehicle', linestyle = '-', color = 'dodgerblue', marker='.', capsize=3, ms=ms)


for i in range(x_proposed.shape[0]):
    ax.text(x_proposed[i]+hp, y_proposed[i]-hp, t_max_list[i], ha='center', va='top', color='C2')
    ax.text(x_proposed_cv[i]-1.5*hp, y_proposed_cv[i]-0.8*hp, t_max_list[i], ha='right', va='center', color='dodgerblue')


plt.legend(loc='upper right')
ax.set_xlabel('Average pedestrian delay (s)')
ax.set_ylabel('Average vehicle delay (s)')
ax.set_yticles(15, 17)
ax.set_xticles(10, 35)
# ax.axis('equal')
fig.set_tight_layout(0.1)
