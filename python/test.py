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

loop_results4, light_results4, ped_results4, band_overlap_list = unit_run(param, signal_control_type="unsynchronized",
                                                    crossing_control_type="proposed",
                                                    pph=40, seed=8, trajectory='trajectory_proposed_un.xml')


# plot figures for proposed unsynchronized
fig4, ax4= plot_trajectory(base_path+'trajectory_proposed_un.xml',
                light_results4, [5220, 3610], direction='EW', ped_results=ped_results4)
plt.xlabel('Time in simulation (s)',fontsize=12)
ax4.set_yticks([250, -103, -450])
ax4.set_yticklabels(['INT 1', 'MID C', 'INT 2'])
ax4.grid(False)
fig4.set_tight_layout(0.1)
fig4.savefig('..//paper//fig_save_3_extended_net_cv//trajectory_proposed_EW_40_original.png', dpi=200)