from functions import *

# MFR03
param = dict(
    delta_t_min=4,
    delta_t_max=40,
    gamma=0,
    v=13.5)

# traci.start(sumoCmd)
# loop_results, light_results = run(param)
#
# # An example PCD of intersection T01
# LP17_data = pd.concat([loop_result_to_dataframe(loop_results['LP17_0']),
#                   loop_result_to_dataframe(loop_results['LP17_1'])])
# T01_data = light_result_to_dataframe(light_results['T01'], 3, 0, 2)
# plot_pcd(T01_data, LP17_data)
#
# # An example PCD of intersection T02 (East to West)
# LP18_data_ = pd.concat([loop_result_to_dataframe(loop_results['-LP18_0']),
#                   loop_result_to_dataframe(loop_results['-LP18_1'])])
# T02_data = light_result_to_dataframe(light_results['T02'], 1, 2, 0)
# plot_pcd(T02_data, LP18_data_)

# %% Analyze the impact of passenger demand
# pph_list = np.arange(5, 125, 15)
pph_list = [50]
# control_types = ["Pelican", "proposed", ""]
# control_types = ["unsignalized"]

# control_types = ["Pelican", "proposed", "unsignalized"]
control_types = ["proposed"]
# Simulate and save results
for pph in pph_list:
    for control_type in control_types:

        loop_results, light_results = unit_run(param, control_type, pph, seed=2)

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

        # save data
        travel_time_data1.to_csv('..//results//TT_WE_pph{}_{}.csv'.format(pph, control_type))
        travel_time_data2.to_csv('..//results//TT_EW_pph{}_{}.csv'.format(pph, control_type))
        travel_time_data.to_csv('..//results//TT_TD_pph{}_{}.csv'.format(pph, control_type))

# Read results and analyze
results = {'Pelican':[], 'proposed':[], 'unsignalized':[]}
E_TT = 50
for control_type in control_types:
    for pph in pph_list:
        TT = pd.read_csv('..//results//TT_EW_pph{}_{}.csv'.format(pph, control_type), index_col=0).values
        results[control_type].append((TT - E_TT).mean())
        # results[signal_control_type].append(TT.sum())

fig, ax = plt.subplots()
for key, value in results.items():
    ax.plot(pph_list, value,'^-', label=key)
plt.legend()
plt.xlabel('Pedestrian Demand (pph)')
plt.ylabel('Average Vehicle Delay (s)')


# %% Travel time histogram analysis
# For east to west direction
TT1 = pd.read_csv('..//results//TT_WE_pph35_Pelican.csv', index_col=0).values
TT2 = pd.read_csv('..//results//TT_WE_pph35_proposed.csv', index_col=0).values

plt.figure()
x1 = np.sort(TT1.ravel())
y1 = np.arange(len(TT1))/float(len(TT1))
x2 = np.sort(TT2.ravel())
y2 = np.arange(len(TT2))/float(len(TT2))
plt.hist(TT1, bins=20, label="Pelican")
plt.hist(TT2, bins=20, label="proposed")
plt.legend(loc='upper left')
plt.xlabel('Travel Time (s)')
plt.ylabel('Frequency of Vehicles')

# %% Travel time cumulative distribution function
# For east to west direction
plt.figure()
plt.plot(x1,y1, label="Pelican")
plt.plot(x2,y2, label="proposed")
plt.legend(loc='upper left')
plt.xlabel('Travel Time (s)')
plt.ylabel('Cumulative Frequency')

# For two directions
TT3 = pd.read_csv('..//results//TT_TD_pph35_Pelican.csv', index_col=0).values
TT4 = pd.read_csv('..//results//TT_TD_pph35_proposed.csv', index_col=0).values
plt.figure()
x3 = np.sort(TT3.ravel())
y3 = np.arange(len(TT3))/float(len(TT3))
x4 = np.sort(TT4.ravel())
y4 = np.arange(len(TT4))/float(len(TT4))
plt.plot(x3,y3, label="Pelican")
plt.plot(x4,y4, label="proposed")
plt.legend(loc='upper left')
plt.xlabel('Travel Time (s)')
plt.ylabel('Cumulative Frequency')



