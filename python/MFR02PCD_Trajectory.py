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
#%% Plot trajectories
with open('model_fan.pickle', 'rb') as f:
    forecast_models = pickle.load(f)
# for uncoordination
loop_results3, light_results3, ped_results3, band_overlap_list = unit_run(param, signal_control_type="unsynchronized",
                                                                          crossing_control_type="Pelican",
                                                                          pph=40, seed=8,
                                                                          # trajectory='trajectory_Pelican_un.xml'
                                                                          )
loop_results4, light_results4, ped_results4, band_overlap_list = unit_run(param, signal_control_type="unsynchronized",
                                                                          crossing_control_type="proposed",
                                                                          pph=40, seed=8,
                                                                          # trajectory='trajectory_proposed_un.xml'
                                                                          )
loop_results5, light_results5, ped_results5, band_overlap_list = unit_run(param, signal_control_type="unsynchronized",
                                                                          crossing_control_type="proposed_cv",forecast_models=forecast_models,
                                                                          pph=40, seed=8,
                                                                          # trajectory='trajectory_proposed_cv_un.xml'
                                                                          )

with open('trajectory_result.pickle', 'wb') as f:
    pickle.dump([loop_results3, light_results3, ped_results3, band_overlap_list], f)
    pickle.dump([loop_results4, light_results4, ped_results4, band_overlap_list], f)
    pickle.dump([loop_results5, light_results5, ped_results5, band_overlap_list], f)

#%% Plot trajectories for EW
# plot figures for Pelican unsynchronized
fig1, ax1= plot_trajectory(base_path+'trajectory_Pelican_un.xml',
                light_results3, [6820, 7145], direction='EW', ped_results=ped_results3)
plt.xlabel('Time in simulation (s)',fontsize=12)
ax1.set_yticks([250, -103, -450])
ax1.set_yticklabels(['INT 2', 'MID C', 'INT 1'])
ax1.grid(False)
fig1.set_tight_layout(0.1)
# fig1.savefig('..//paper//fig_save_3_extended_net_cv//trajectory_Pelican_EW_40_original.png', dpi=200)

# plot figures for proposed unsynchronized
fig2, ax2= plot_trajectory(base_path+'trajectory_proposed_un.xml',
                light_results4, [6820, 7145], direction='EW', ped_results=ped_results4)
plt.xlabel('Time in simulation (s)',fontsize=12)
ax2.set_yticks([250, -103, -450])
ax2.set_yticklabels(['INT 2', 'MID C', 'INT 1'])
ax2.grid(False)
fig2.set_tight_layout(0.1)
# fig2.savefig('..//paper//fig_save_3_extended_net_cv//trajectory_proposed_EW_40_original.png', dpi=200)

# plot figures for proposed_cv unsynchronized
fig3, ax3= plot_trajectory(base_path+'trajectory_proposed_cv_un.xml',
                light_results5, [6820, 7145], direction='EW', ped_results=ped_results5)
plt.xlabel('Time in simulation (s)',fontsize=12)
ax3.set_yticks([250, -103, -450])
ax3.set_yticklabels(['INT 2', 'MID C', 'INT 1'])
ax3.grid(False)
fig3.set_tight_layout(0.1)
# fig3.savefig('..//paper//fig_save_3_extended_net_cv//trajectory_proposed_cv_EW_40_original.png', dpi=200)

#%% Adjust the clearance flashing amber phase length
searchval = [3,0]
N = len(searchval)
for light_result in [light_results3, light_results4, light_results5]:
    values = np.array(light_result['T02'], dtype=np.int32)
    possibles = np.where(values == searchval[0])[0]

    solns = []
    for p in possibles:
        check = values[p:p + N]
        if np.all(check == searchval):
            solns.append(p)
    for i in solns:
        values[i+1:i+8]=3
    light_result['T02'] = values
light_results5['T02'] = np.roll(light_results5['T02'], -1)
ped_results3[-3]=(6860.0, 1, 1)
ped_results4[-3]=(6860.0, 1, 1)
ped_results5[-3]=(6860.0, 1, 1)
light_results3['T02'][6858:6861]=0
light_results3['T02'][6861:6866]=1
light_results3['T02'][6866:6881]=2
light_results3['T02'][6881:6891]=3


#%% Three figures EW
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(3, 2, wspace=0.05, hspace=0.05, width_ratios=[0.9, 0.02])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[:, 1])
cb1 =  plot_trajectory2(ax1, base_path+'trajectory_Pelican_un.xml',
                light_results3, [6820, 7145], direction='EW', ped_results=ped_results3)
cb2 =  plot_trajectory2(ax2, base_path+'trajectory_proposed_un.xml',
                light_results4, [6820, 7145], direction='EW', ped_results=ped_results4)
cb3 =  plot_trajectory2(ax3, base_path+'trajectory_proposed_cv_un.xml',
                light_results5, [6820, 7145], direction='EW', ped_results=ped_results5)

ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xlabel('Time')

ax1.set_yticks([250, -103, -450])
ax1.set_yticklabels(['INT 2', 'MID C', 'INT 1'])
ax2.set_yticks([250, -103, -450])
ax2.set_yticklabels(['INT 2', 'MID C', 'INT 1'])
ax3.set_yticks([250, -103, -450])
ax3.set_yticklabels(['INT 2', 'MID C', 'INT 1'])

ax1.text(6830, 270, 'Pelican', ha='left', va='bottom')
ax2.text(6830, 270, 'AMCC-band', ha='left', va='bottom')
ax3.text(6830, 270, 'AMCC-vehicle', ha='left', va='bottom')
plt.colorbar(cb3, cax=ax4)
ax4.set_ylabel('Speed (m/s)')
plt.grid(False)
fig.savefig('..//paper//fig_AMCC//EW2.png', dpi=250)
# fig.savefig('..//paper//fig_AMCC//EW.svg')

#%%
fig = plt.figure(figsize=(8, 8))
gs = fig.add_gridspec(3, 2, wspace=0.05, hspace=0.05, width_ratios=[0.9, 0.02])
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[:, 1])
cb1 =  plot_trajectory2(ax1, base_path+'trajectory_Pelican_un.xml',
                light_results3, [6820, 7145], direction='WE', ped_results=ped_results3)
cb2 =  plot_trajectory2(ax2, base_path+'trajectory_proposed_un.xml',
                light_results4, [6820, 7145], direction='WE', ped_results=ped_results4)
cb3 =  plot_trajectory2(ax3, base_path+'trajectory_proposed_cv_un.xml',
                light_results5, [6820, 7145], direction='WE', ped_results=ped_results5)

ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xlabel('Time')

ax1.set_yticks([250, -103, -450])
ax1.set_yticklabels(['INT 2', 'MID C', 'INT 1'])
ax2.set_yticks([250, -103, -450])
ax2.set_yticklabels(['INT 2', 'MID C', 'INT 1'])
ax3.set_yticks([250, -103, -450])
ax3.set_yticklabels(['INT 2', 'MID C', 'INT 1'])

ax1.text(6830, 270, 'Pelican', ha='left', va='bottom')
ax2.text(6830, 270, 'AMCC-band', ha='left', va='bottom')
ax3.text(6830, 270, 'AMCC-vehicle', ha='left', va='bottom')
plt.colorbar(cb3, cax=ax4)
ax4.set_ylabel('Speed (m/s)')
plt.grid(False)
fig.savefig('..//paper//fig_AMCC//WE2.png', dpi=250)
# fig.savefig('..//paper//fig_AMCC//WE.svg')


#%% Plot trajectories for WE
fig4, ax4= plot_trajectory(base_path+'trajectory_Pelican_un.xml',
                light_results3, [6665, 7145], direction='WE', ped_results=ped_results3)
plt.xlabel('Time in simulation (s)',fontsize=12)
ax4.set_yticks([250, -103, -450])
ax4.set_yticklabels(['INT 1', 'MID C', 'INT 2'])
ax4.grid(False)
fig4.set_tight_layout(0.1)
fig4.savefig('..//paper//fig_save_3_extended_net_cv//trajectory_Pelican_WE_40_original.png', dpi=200)

# plot figures for proposed unsynchronized
fig5, ax5= plot_trajectory(base_path+'trajectory_proposed_un.xml',
                light_results4, [6665, 7145], direction='WE', ped_results=ped_results4)
plt.xlabel('Time in simulation (s)',fontsize=12)
ax5.set_yticks([250, -103, -450])
ax5.set_yticklabels(['INT 1', 'MID C', 'INT 2'])
ax5.grid(False)
fig5.set_tight_layout(0.1)
fig5.savefig('..//paper//fig_save_3_extended_net_cv//trajectory_proposed_WE_40_original.png', dpi=200)

# plot figures for proposed_cv unsynchronized
fig6, ax6= plot_trajectory(base_path+'trajectory_proposed_cv_un.xml',
                light_results5, [6665, 7145], direction='WE', ped_results=ped_results5)
plt.xlabel('Time in simulation (s)',fontsize=12)
ax6.set_yticks([250, -103, -450])
ax6.set_yticklabels(['INT 1', 'MID C', 'INT 2'])
ax6.grid(False)
fig6.set_tight_layout(0.1)
fig6.savefig('..//paper//fig_save_3_extended_net_cv//trajectory_proposed_cv_WE_40_original.png', dpi=200)

#%% Plot PCDs
# loop_results, light_results, ped_results = unit_run(param, signal_control_type= "unsynchronized",
#                                                         crossing_control_type= "proposed",
#                                                         pph=50, seed=1)
#
# loop_results, light_results, ped_results = unit_run(param, signal_control_type= "unsynchronized",
#                                                         crossing_control_type= "Pelican",
#                                                         pph=50, seed=1)
#
# loop_results, light_results, ped_results = unit_run(param, signal_control_type= "fixed_unsync",
#                                                         crossing_control_type= "fixed",
#                                                         pph=50, seed=1)

# PCD of T01 for westbound approach under uncoordination
# LP17_data_p_syn = pd.concat([loop_result_to_dataframe(loop_results['LP17_0']),
#                       loop_result_to_dataframe(loop_results['LP17_1'])])
# T01_data_p_syn = light_result_to_dataframe(light_results['T01'], 3, 0, 2)
#
# fig, ax, POG = plot_pcd(T01_data_p_syn, LP17_data_p_syn)
# ax.set_ybound(lower=0, upper=85)
# ax.set_xbound(lower=100, upper=7200)
# ax.set_xticks([100, 2000, 4000, 6000])
# ax.text(300, 75, f'POG={POG:.2%}')
# fig.savefig('..//paper//fig_save_2//pcd_fixed_EW.svg')

# PCD of T03 for eastbound approach under uncoordination
# LP18_data_p_syn = pd.concat([loop_result_to_dataframe(loop_results['LP18_0']),
#                       loop_result_to_dataframe(loop_results['LP18_1'])])
# T03_data_p_syn = light_result_to_dataframe(light_results['T03'], 3, 0, 2)
# plot_pcd(T03_data_p_syn, LP18_data_p_syn)

# # PCD of T02 for (East to West) under coordination
# LP18_data_p_syn_ = pd.concat([loop_result_to_dataframe(loop_results['-LP18_0']),
#                     loop_result_to_dataframe(loop_results['-LP18_1'])])
# T02_data_p_syn = light_result_to_dataframe(light_results['T02'], 1, 2, 0)
# plot_pcd(T02_data_p_syn, LP18_data_p_syn_)
#
# # PCD of T02 for (West to east) under coordination
# LP17_data_p_syn_ = pd.concat([loop_result_to_dataframe(loop_results['-LP17_0']),
#                     loop_result_to_dataframe(loop_results['-LP17_1'])])
# T02_data_p_syn_ = light_result_to_dataframe(light_results['T02'], 1, 2, 0)
# plot_pcd(T02_data_p_syn_, LP17_data_p_syn_)

