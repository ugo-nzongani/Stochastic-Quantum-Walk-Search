from qswsearch import *
import csv
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import argrelextrema
import numpy as np

# Set the global font to be the LaTeX font
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
})

def save_data(qsw,name):
    # Sample data as a list of lists
    data = [
        ['Omega','Gamma','Marked_node','Sink_rate'],
        [qsw.w,qsw.gamma,qsw.marked_node,qsw.sink_rate]
    ]
    data.append('')
    data.append(['Time', 'Success'])
    for i in range(len(qsw.time_list)):
        data.append([qsw.time_list[i],qsw.success[i]])
    last_index = len(data)+2
    data.append('')
    data.append(['Adjacency'])
    for i in range(qsw.n_nodes):
        data.append([])
        for j in range(qsw.n_nodes):
            data[last_index].append(qsw.adjacency[i][j])
        last_index += 1

    # Writing data to a CSV file
    with open(name+'.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

def read_csv(file):
    df = pd.read_csv(file+'.csv',on_bad_lines='skip')
    columns_dict = {col: df[col].tolist() for col in df.columns}
    rows_with_value_time = df[df['Omega'] == 'Time']
    row_numbers_time = rows_with_value_time.index.tolist()[0]
    rows_with_value_adjacency = df[df['Omega'] == 'Adjacency']
    row_numbers_adjacency = rows_with_value_adjacency.index.tolist()[0]

    w = float(df['Omega'][0])
    if w == 1.0:
        gamma = 0.
    else:
        gamma = float(df['Gamma'][0])
    marked_node = float(df['Marked_node'][0])
    sink_rate = float(df['Sink_rate'][0])
    time = [float(t) for t in columns_dict['Omega'][row_numbers_time+1:row_numbers_adjacency]]
    prob = [float(p) for p in columns_dict['Gamma'][row_numbers_time+1:row_numbers_adjacency]]

    dt = time[1]-time[0]

    df = pd.read_csv(file+'.csv',delimiter='\t')
    values = df[df.columns[0]].tolist()
    index_of_adjacency = values.index('Adjacency')
    values[index_of_adjacency+1]
    adjacency = []
    for row in values[index_of_adjacency+1:]:
        str_list = row.split(',')
        int_list = [int(num) for num in str_list]
        adjacency.append(int_list)
    res = {}
    res['w'] = w
    res['gamma'] = gamma
    res['marked_node'] = marked_node
    res['sink_rate'] = sink_rate
    res['time'] = time
    res['prob'] = prob
    res['dt'] = dt
    res['adjacency'] = np.array(adjacency)
    return res

def get_data(file,w_list=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]):
    #w_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    data = []
    for w in w_list:
        name_w = file+str(w)
        data.append(read_csv(name_w))
    return data

def multiple_plot(data,save=False,name='plot',dpi=300):
    w_list = [data[i]['w'] for i in range(len(data))]
    # Create 3x4 grid of subplots
    fig, axs = plt.subplots(3, 4, figsize=(12, 9), constrained_layout=False)

    # Remove the last subplot
    fig.delaxes(axs[2, 3])

    # Flatten the axs array for easy iteration
    axs = axs.flatten()

    cmap = LinearSegmentedColormap.from_list("blue_red", ["blue", "red"], N=len(w_list))

    # Plot each curve in a separate subplot
    for i in range(len(w_list)):
        time = data[i]['time']
        prob = data[i]['prob']
        gamma = data[i]['gamma']
        color = cmap(i / (len(w_list) - 1))
        axs[i].plot(time, prob, color=color, label=f'$\\gamma$={gamma}')
        #axs[i].grid()
        axs[i].set_title(f'$\omega=${w_list[i]}')
        axs[i].legend()

    # Get the position of the removed subplot
    bbox = axs[-1].get_position()

    # Manually add legend for a specific subplot and move it to the lower right of the removed subplot
    #axs[0].legend(loc='lower right', bbox_to_anchor=(bbox.x1, bbox.y0), bbox_transform=plt.gcf().transFigure)

    # Add global labels
    plt.figtext(0.5, 0.06, 'Time', ha='center', fontsize=12)
    plt.figtext(0.06, 0.5, 'Cost function $f_{\omega}$', va='center', rotation='vertical', fontsize=12)

    # Adjust layout and show plot
    if save:
        plt.savefig(name+'.png', dpi=dpi)
    plt.show()
    
def efficiency_plot(data,label='label',marker_list='o',save=False,name='plot',dpi=300):
    if isinstance(data[0], dict):
        data = [data]
    if not isinstance(label,list):
        label = [label]
    if not isinstance(marker_list,list):
        marker_list = [marker_list]
    if len(label) != len(data):
        for i in range(len(data)-len(label)):
            label.append('')
    if len(marker_list) != len(data):
        for i in range(len(data)-len(marker_list)):
            marker_list.append('o')
    w_list = [data[0][i]['w'] for i in range(len(data[0]))]
    blue_to_red = LinearSegmentedColormap.from_list('blue_to_red', ['blue', 'red'])
    for k,d in enumerate(data):
        integral = []
        gamma = []
        for i in range(len(w_list)):
            a = d[i]['time'][0]
            b = d[i]['time'][-1]
            dt = d[i]['dt']
            x = [d[i]['time'][j] for j in range(len(d[i]['time']))] #/d[i]['time'][-1]
            y = d[i]['prob']
            integral.append(np.trapz(y,x,dt)/d[i]['time'][-1])
            gamma.append(d[i]['gamma'])
        plt.scatter(w_list,integral,c=gamma,label=label[k],marker=marker_list[k],cmap='gist_rainbow')
        plt.legend()
    cbar = plt.colorbar()
    cbar.set_label('$\gamma^*$', fontsize=14)
    plt.xticks(w_list)
    plt.xlabel('$\omega$',fontsize=14)
    plt.ylabel('$\Psi(\omega,t)$',fontsize=14)
    #plt.title(name)
    if save:
        plt.savefig(name+'.png', dpi=dpi)
    plt.show()
    
def efficiency(data,max_time=-1):
    dt = data['dt']
    if max_time == -1:
        max_time = data['time'][-1]
    max_time_index = data['time'].index(max_time)
    x = data['time'][:max_time_index] #[data['time'][j] for j in range(len(data['time'][:]))]
    y = data['prob'][:max_time_index]
    return np.trapz(y,x,dt)/data['time'][-1]

def mean_efficiency(list_of_data):
    mean_prob = []
    mean_gamma = []
    w_list = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
    for w in range(len(w_list)):
        prob = []
        gamma = []
        for i in range(len(list_of_data)):
            prob.append(efficiency(list_of_data[i][w]))
            gamma.append(list_of_data[i][w]['gamma'])
        mean_prob.append(np.mean(prob))
        mean_gamma.append(np.mean(gamma))
    return mean_prob,mean_gamma

# QSW Search (QS)
'''
def is_oscillating(data):
    """
    Determines if the given data array has oscillating values.

    Parameters:
    data (numpy.ndarray): The input data array.

    Returns:
    bool: True if the data has oscillating values, False otherwise.
    """
    # Compute the numerical first derivative
    diffs = np.diff(data)
    
    # Check if the differences change sign
    signs = np.sign(diffs)
    sign_changes = np.diff(signs)
    
    # Oscillating if there are both positive and negative sign changes
    has_positive_change = np.any(sign_changes > 0)
    has_negative_change = np.any(sign_changes < 0)
    
    return has_positive_change and has_negative_change
'''

def is_oscillating(curve, threshold=3, tolerance=1e-5):
    """
    Determines if a given list represents an oscillating curve.

    Parameters:
    - curve: list of numerical values representing the curve.
    - threshold: minimum number of direction changes to be considered oscillating.
    - tolerance: small value to ignore tiny fluctuations.

    Returns:
    - True if the curve is oscillating, False otherwise.
    """
    # Calculate the differences between consecutive elements
    changes = []
    for i in range(1, len(curve)):
        diff = curve[i] - curve[i - 1]
        if abs(diff) > tolerance:  # Ignore changes smaller than the tolerance
            changes.append(diff)

    # Count the number of direction changes (ignoring small changes)
    direction_changes = 0
    for i in range(1, len(changes)):
        # Check if the direction changes (sign flips)
        if changes[i] * changes[i - 1] < 0:
            direction_changes += 1

    # Return True if the number of direction changes meets the threshold
    return direction_changes >= threshold

def comp(x,y):
    res = [False] * len(x)
    threshold = 1e-2
    for i in range(len(x)):
        if abs(x[i]-max(y)) < threshold:
            res[i] = True
    return res

def search_performances(data):
    x = np.array(data['prob'])
    if is_oscillating(x):
        #print('oscillating')
        comp_fun = np.greater
    else:
        #print('not oscillating')
        comp_fun = comp #np.greater
    return argrelextrema(x, comp_fun)