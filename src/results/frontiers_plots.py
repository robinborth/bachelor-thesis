from src.mbd.LogAnalyzer import DataflashLog
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from src.utils import load_graph

def ulg2df_single_flight(file):
    data = DataflashLog.DataflashLog(file)
    comps = {}
    for comp, comp_value in data.channels.items():
        frame = pd.DataFrame(columns=comp_value.keys())
        for k, v in comp_value.items():
            frame[k] = pd.Series(v.dictData)
        comps.update({comp: frame})
    comps.update({'frame': data.frame})
    comps.update({'MSG': data.messages})
    comps.update({'param': data.parameters})
    comps.update({'vehicleType': data.vehicleTypeString})
    comps.update({'mode': data.modeChanges})
    return comps

def plot_flight_signal(df, comp_name, signal_name):
    if 'TimeUS' in df[comp_name]:
        start_time = df[comp_name][signal_name].iloc[0]
        plt.plot((df[comp_name]['TimeUS'] - start_time) / 1000000, df[comp_name][signal_name].values)
        plt.show()
    elif 'TimeMS' in df[comp_name]:
        start_time = df[comp_name][signal_name].iloc[0]
        plt.plot((df[comp_name]['TimeMS'] - start_time) / 1000, df[comp_name][signal_name].values)
        plt.show()

def plot_4_graphs():
    g = load_graph("baseline")
    parents = nx.ancestors(g,'crash')
    nx.draw(g, pos=nx.circular_layout(g), with_labels=True, arrowsize=20, node_color='darkgrey')
    plt.show()
    nx.draw_circular(g, with_labels=True, arrowsize=20, labels=g.nodes, node_color='darkgrey')
    plt.show()

nodes_in_intersction_polished = {
        'ERR_10_5':'loiter',
        'EV_30':'autotune',
        'ERR_18_2':'baro',
        'EV_15':'set_home',
        'EV_29':'super simple',
        'ERR_6_1':'battery failsafe',
        'EV_10':'armed',
        'ERR_7_1':'gps failsafe',
        'EV_62':'yaw reset',
        'ERR_12_1':'crash',
        'EV_18':'land',
        'ERR_16_2':'ekf',
        'ERR_11_2':'gps',
        'EV_11':'disarmed',
        'ERR_5_1':'radio failsafe',
        'EV_57':'motors',
        'ERR_17_1':'ekf failsafe',
        'ERR_24_1': 'ekf change'}

def map_events(incidents):
    incidntsCopy = incidents.copy()
    for k,v in incidents.items():
            if v in nodes_in_intersction_polished:
                nodes_in_intersction_polished[v] = nodes_in_intersction_polished[v].replace(' ','_')
                incidents[k] = 'Exo_'+nodes_in_intersction_polished[v]
            else:
                incidents = incidents.drop(k)
    return incidents

def extract_variables_from_logs(log):
    """ extracts ERR and EV from a chunk of .log files"""
    log = ulg2df_single_flight(log)
    discrete_events = []
    for comp_name, comp_df in log.items():
        if 'EV' == comp_name:
            for event_time, event_id in comp_df['Id'].iteritems():
                discrete_events.append('EV_'+str(event_id))
        if 'ERR' == comp_name:
            for error_time, error_row in  comp_df.iterrows():
                discrete_events.append('ERR_'+str(error_row['Subsys'])+'_'+str(error_row['ECode']))
    return pd.Series(list(set(discrete_events)))


def plot_alt_log1(log):
    df = log
    start_time = df['GPS']['TimeUS'].iloc[0]
    start_altitude = df['GPS']['Alt'].iloc[0]
    min_alt = (df['GPS']['Alt'].values - start_altitude).min()
    max_alt = (df['GPS']['Alt'].values - start_altitude).max()
    fig, ax = plt.subplots(figsize=(18, 5))
    event_height = 20
    ax.set_xlabel('Time (S)', size = 15)
    ax.set_ylabel('Altitude (m)', size = 15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    # signal
    plt.plot((df['GPS']['TimeUS'] - start_time) / 1000000, df['GPS']['Alt'].values - start_altitude)

    # crash event
    x_crash_time = (log['ERR']['TimeUS'].iloc[-1] - start_time) / 1000000
    ax.text(x_crash_time, event_height+5, 'Crash', rotation=90, size = 15)
    ax.vlines(x=x_crash_time, ymin=min_alt, ymax=event_height, colors='green', ls=':', lw=2, label='vline_single - partial height')
    ax.plot(x_crash_time, event_height, marker=10, color = 'g')

    # position hold event
    x_position_hold_time =  (df['GPS'][df['GPS'].index>62645]['TimeUS'].iloc[0] - start_time)/1000000
    ax.text(x_position_hold_time, event_height+5, 'Position hold', rotation=90, size = 11)
    ax.vlines(x=x_position_hold_time, ymin=min_alt, ymax=event_height, colors='green', ls=':', lw=2, label='vline_single - partial height')
    ax.plot(x_position_hold_time, event_height, marker=10, color = 'g')

    # ekf error
    x_position_ekf =  (log['ERR']['TimeUS'].iloc[0] - start_time) / 1000000
    ax.text(x_position_ekf, event_height+5, 'ekf', rotation=90, size = 15)
    ax.vlines(x=x_position_ekf, ymin=min_alt, ymax=event_height, colors='green', ls=':', lw=2, label='vline_single - partial height')
    ax.plot(x_position_ekf, event_height, marker=10, color = 'g')

    # plot generics
    ax.set_xlabel('Time (S)', size = 15)
    ax.set_ylabel('Altitude (m)', size = 15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    plt.ylim([min_alt, max_alt + 5])

    plt.show()
    fig.savefig('log1.pdf')

def plot_alt_log2(log):
    df = log
    start_time = df['GPS']['TimeUS'].iloc[0]
    start_altitude = df['GPS']['Alt'].iloc[0]
    min_alt = (df['GPS']['Alt'].values - start_altitude).min()
    max_alt = (df['GPS']['Alt'].values - start_altitude).max()
    fig, ax = plt.subplots(figsize=(18, 5))
    event_height = 6
    ax.set_xlabel('Time (S)', size = 15)
    ax.set_ylabel('Altitude (m)', size = 15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    # signal
    plt.plot((df['GPS']['TimeUS'] - start_time) / 1000000, df['GPS']['Alt'].values - start_altitude)

    # crash event
    x_crash_time = (log['ERR']['TimeUS'].iloc[-1] - start_time) / 1000000
    plt.text(x_crash_time, event_height+1, 'Crash', rotation=90, size = 15)
    plt.vlines(x=x_crash_time, ymin=min_alt, ymax=event_height, colors='green', ls=':', lw=2, label='vline_single - partial height')
    plt.plot(x_crash_time, event_height, marker=10, color = 'g')

    # battery fs event
    x_batt_fs_ekf =  (log['ERR']['TimeUS'].iloc[-3] - start_time) / 1000000
    plt.text(x_batt_fs_ekf, event_height+1, 'battery failsafe', rotation=90, size = 11)
    plt.vlines(x=x_batt_fs_ekf, ymin=min_alt, ymax=event_height, colors='green', ls=':', lw=2, label='vline_single - partial height')
    plt.plot(x_batt_fs_ekf, event_height, marker=10, color = 'g')

    # ekf error
    x_position_ekf =  (log['ERR']['TimeUS'].iloc[-2] - start_time) / 1000000
    plt.text(x_position_ekf, event_height+1, 'ekf', rotation=90, size = 15)
    plt.vlines(x=x_position_ekf, ymin=min_alt, ymax=event_height, colors='green', ls=':', lw=2, label='vline_single - partial height')
    plt.plot(x_position_ekf, event_height, marker=10, color = 'g')

    plt.ylim([0, 10])
    plt.xlim([685, 805])
    plt.show()
    fig.savefig('log2.pdf')

def plot_alt_log2_with_break(log):
    df = log
    start_time = df['GPS']['TimeUS'].iloc[0]
    start_altitude = df['GPS']['Alt'].iloc[0]
    min_alt = (df['GPS']['Alt'].values - start_altitude).min()
    max_alt = (df['GPS']['Alt'].values - start_altitude).max()
    fig, (ax,ax2) = plt.subplots(1,2,figsize=(18, 5))
    event_height = 70
    ax.set_xlabel('Time (S)', size=15)
    ax.set_ylabel('Altitude (m)', size=15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    # signal
    plt.plot((df['GPS']['TimeUS'] - start_time) / 1000, df['GPS']['Alt'].values - start_altitude)

    # signal
    x = (df['GPS']['TimeUS'] - start_time) / 1000
    y = df['GPS']['Alt'].values - start_altitude
    ax.plot(x, y)
    ax2.plot(x, y)
    ax.set_xlim(0, 230)
    ax2.set_xlim(600, 830)

    ## hide the spines between ax and ax2
    ax.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax.yaxis.tick_left()
    ax.tick_params(labelright='off')
    ax2.yaxis.tick_right()

    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    ax.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax2.plot((-d, +d), (-d, +d), **kwargs)

    # crash event
    x_crash_time = (log['ERR']['TimeUS'].iloc[-1] - start_time) / 1000
    plt.text(x_crash_time, event_height + 5, 'Crash', rotation=90, size=15)
    plt.vlines(x=x_crash_time, ymin=min_alt, ymax=event_height, colors='green', ls=':', lw=2,
               label='vline_single - partial height')
    plt.plot(x_crash_time, event_height + 2.5, marker=10, color='g')

    # battery fs event
    x_batt_fs_ekf = (log['ERR']['TimeUS'].iloc[-3] - start_time) / 1000
    plt.text(x_batt_fs_ekf, event_height + 5, 'battery failsafe', rotation=90, size=15)
    plt.vlines(x=x_batt_fs_ekf, ymin=min_alt, ymax=event_height, colors='green', ls=':', lw=2,
               label='vline_single - partial height')
    plt.plot(x_batt_fs_ekf, event_height, marker=10, color='g')

    # ekf error
    x_position_ekf = (log['ERR']['TimeUS'].iloc[-2] - start_time) / 1000
    plt.text(x_position_ekf, event_height + 5, 'ekf', rotation=90, size=15)
    plt.vlines(x=x_position_ekf, ymin=min_alt, ymax=event_height, colors='green', ls=':', lw=2,
               label='vline_single - partial height')
    plt.plot(x_position_ekf, event_height, marker=10, color='g')

    plt.ylim([min_alt, max_alt + 5])
    plt.show()
    fig.savefig('log1.pdf')

def plot_alt_log3(log):
    df = log
    start_time = df['GPS']['TimeMS'].iloc[0]
    start_altitude = df['GPS']['Alt'].iloc[0]
    min_alt = (df['GPS']['Alt'].values - start_altitude).min()
    max_alt = (df['GPS']['Alt'].values - start_altitude).max()
    fig, ax = plt.subplots(figsize=(18, 5))
    event_height = 6
    ax.set_xlabel('Time (S)', size = 15)
    ax.set_ylabel('Altitude (m)', size = 15)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)

    # signal
    plt.plot((df['GPS']['TimeMS'] - start_time) / 1000, df['GPS']['Alt'].values - start_altitude)

    # crash event
    x_crash_time = (109324000 - start_time)/1000
    plt.text(x_crash_time, event_height+15, 'Crash', rotation=90, size = 15)
    plt.vlines(x=x_crash_time, ymin=min_alt, ymax=event_height, colors='green', ls=':', lw=2, label='vline_single - partial height')
    plt.plot(x_crash_time, event_height, marker=10, color = 'g')

    # radio fs event
    x_radio_fs_ekf =  (109305000 - start_time)/1000
    plt.text(x_radio_fs_ekf, event_height+15, 'radio failsafe and rtl', rotation=90, size = 15)
    plt.vlines(x=x_radio_fs_ekf, ymin=min_alt, ymax=event_height, colors='green', ls=':', lw=2, label='vline_single - partial height')
    plt.plot(x_radio_fs_ekf, event_height, marker=10, color = 'g')

    # rtl
    # x_position_ekf =  (109304000 - start_time)/1000
    # plt.text(x_position_ekf, event_height+5, 'rtl', rotation=90, size = 15)
    # plt.vlines(x=x_position_ekf, ymin=min_alt, ymax=event_height, colors='green', ls=':', lw=2, label='vline_single - partial height')
    # plt.plot(x_position_ekf, event_height, marker=10, color = 'g')

    # plt.ylim([0, 10])
    # plt.xlim([685, 805])
    plt.show()
    fig.savefig('log3.pdf')

# TODO: run HP2SAT from python
# import os.path,subprocess
# from subprocess import STDOUT,PIPE
#
# def compile_java(java_file):
#     subprocess.check_call(['javac', java_file])
#
# def execute_java(java_file):
#     java_class,ext = os.path.splitext(java_file)
#     cmd = ['java', java_class]
#     proc = subprocess.Popen(cmd, stdout=PIPE, stderr=STDOUT)
#     stdout,stderr = proc.communicate(input='SomeInputstring')
#     print ('This was "' + stdout + '"')

if __name__ == "__main__":

    # log_address = '../../src/mbd/LogAnalyzer/examples/fly away in z/downward fall/tracking module problem/mode change/57ff7deebb9f806b20ca8bfe.log'
    # log_address = '../../src/mbd/LogAnalyzer/examples/fly away in z/downward fly/low battery/58f50200b10f754f0769bb2b_suspicous batt.log'
    log_address = '../../src/mbd/LogAnalyzer/examples/mechanical_fail.log'


    log = ulg2df_single_flight(log_address)
    plot_alt_log1(log)
    plot_alt_log2(log)
    plot_alt_log3(log)
    incidents = extract_variables_from_logs(log_address)
    mapped_incidents = map_events(incidents)
    g = load_graph("baseline")
    plot_4_graphs()
