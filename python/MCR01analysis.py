from functions import *

# MFR03
param = dict(
    delta_t_min=4,
    delta_t_max=40,
    gamma=0,
    v=13.89)
        # <route-files value="vehicle.rou.xml, pedestrians.trip.xml"/>

## Output related results: ATT, AVD, APD, PD
loop_results, light_results = unit_run(param, 'Pelican', 20)
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
