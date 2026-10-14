""" This script gets the AP and LAT position of each LFP site and plots it in a 2D space. By default
it takes only the 'good sessions', i.e. those that are used for the laminar analysis.
Comment one or the other monkey to get the cortical map of the corresponding monkey. Then, plot them
together with rotation + translation for getting the combined cortical reconstruction.
"""

import os
import warnings
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# to remove the spam of pandas FutureWarning with iteritems
warnings.simplefilter(action='ignore', category=FutureWarning)

# check where are we running the code
current_path = os.getcwd()

if current_path.startswith('C:'):
    SERVER = 'W:'  # local w VPN

else:
    SERVER = '/envau/work/'

PATH = SERVER + '/comco/lopez.l/Electrophysiology/ephy_laminar_MUA/Raw_data'
file_electrodes_info = os.path.join(PATH, "ephydataset_info_MUA.xlsx")

# read the excel with pandas
df_electrodes = pd.read_excel(file_electrodes_info, sheet_name=0)  # read the excel with pandas
df_electrodes.set_index('SESSION', inplace=True, drop=True)  # get session as the search index

# select the good sessions
good_sessions = df_electrodes.index.values[df_electrodes.USE_NEW == 'y']
probes = [1, 2]

# plot the figure
sns.set(style='ticks')
fig, ax = plt.figure(figsize=(15, 15)), plt.subplot(111)
ax.axvline(-1, color='k', linestyle='--', alpha=.25)
# ax.axvspan(-1, 6, color='lightblue', alpha=.15, label='PMd')
# ax.axvspan(-1, -4.5, color='orange', alpha=.15, label='M1')

ax.set_xlim([-9.5, 9.5])
ax.set_ylim([-9.5, 9.5])

locations = []

monkey = 'M'
MARKER_SHAPE = 's'

for session in good_sessions:
    if session.startswith(monkey):
        for probe in probes:
            if (probe == 2) & (df_electrodes.loc[session]['PROBE_2'] == '-'):
                pass
            else:
                # get the color depending on the beta band dominance
                if df_electrodes.loc[session][f'BB_{probe}'] == 'LD':
                    COLOR = 'orange'
                elif df_electrodes.loc[session][f'BB_{probe}'] == 'HD':
                    COLOR = 'tab:blue'
                else:
                    COLOR = 'gray'

                # get the AP and LAT position of the first probe
                ap = df_electrodes.loc[session][f'AP_{probe}']
                lat = df_electrodes.loc[session][f'LAT_{probe}']

                # check if the location is already in the list - avoid complete overlap in the plot
                while [ap, lat] in locations:
                    ap += .05
                    lat += .1
                locations.append([ap, lat])

                ax.plot(ap, lat, marker=MARKER_SHAPE, markersize=15, c='gray')
                ax.text(ap + .1, lat + .1, f'{session}-{probe}', fontsize=12)
    else:
        continue


ax.invert_xaxis()
ax.legend(loc='upper right')
ax.set_xlabel('AP')
ax.set_ylabel('LAT')
ax.set_title('Electrodes location in the cortex across all LFP sites', fontsize=14, weight='bold')
sns.despine(ax=ax, trim=True)
#fig.savefig(os.path.join(PATH, f'Tomy_electrodes_location_wotxt_new.svg'), dpi=300)
