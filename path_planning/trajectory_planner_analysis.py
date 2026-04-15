import numpy as np
import heapq
from scipy.ndimage import binary_dilation

import matplotlib.pyplot as plt

if __name__ == "__main__":
    '''
    Analysis data format: 
    
    RRT*: [elapsed_time, straight_line_distance, actual_path_length, path_efficiency, max_iterations]
    A*: [elapsed_time, straight_line_distance, actual_path_length, path_efficiency, nodes_expanded, epsilon]
    
    5x point for each list. Take a screen shot of Rviz for each list, so 6x images in total. 

    Order of screenshots:
    Astar epsilon = 1, path 1
    Astar epsilon = 1, path 2
    Astar epsilon = 1.5, path 1
    Astar epsilon = 1.5, path 2
    Astar epsilon = 0.7, path 1
    Astar epsilon = 0.7, path 2

    RRT* max iterations = 10000, path 1
    RRT* max iterations = 10000, path 2
    RRT* max iterations = 15000, path 1
    RRT* max iterations = 15000, path 2
    RRT* max iterations = 7000, path 1
    RRT* max iterations = 7000, path 2
    '''
    
    # RRT* for two paths. Max iterations = 10000, 15000, 7000.
    rrt_star_data_path1_it_10000 = [
        [9.6752,554.8279,643.5941,0.8621,10000],
        [10.4609,557.3912,637.3200,0.8746,10000],
        [10.8551,563.5113,661.4811,0.8519,10000],
        [7.9705,564.3545,670.9623,0.8410,10000],
        [8.7504,548.4706,688.0040,0.7972,10000],        
    ]
    rrt_star_data_path2_it_10000 = [
        [6.6974,672.3764,990.4238,0.6789,10000],
        [7.5918,703.3193,981.5430,0.7165,10000],
        [8.2338,674.9993,971.0711,0.6951,10000],
        [10.3796,669.6066,959.5599,0.6978,10000],
        [5.8295,693.6065,1007.9490,0.6881,10000],        
    ]
    rrt_star_data_path1_it_15000 = [
        [18.7939,554.3014,643.7664,0.8610,15000],
        [17.0710,551.7943,628.0643,0.8786,15000],
        [17.9090,558.8211,618.5830,0.9034,15000],
        [19.8360,552.6346,650.8677,0.8491,15000],
        [20.0620,562.7664,657.8504,0.8555,15000],  
    ]
    rrt_star_data_path2_it_15000 = [
        [13.6602,678.4578,1026.1112,0.6612,15000],
        [13.1314,674.8941,1011.8536,0.6670,15000],
        [22.4306,679.2415,973.7446,0.6976,15000],
        [18.7082,679.6499,932.8333,0.7286,15000],
        [3.1120,675.5302,1036.4881,0.6517,15000],  
        [2.2927,667.6114,1017.6076,0.6561,10000],      
    ]
    rrt_star_data_path1_it_7000 = [
        [6.3376,556.0405,657.8100,0.8453,7000],
        [6.4392,560.0545,693.3904,0.8077,7000],
        [4.7868,553.4799,678.7504,0.8154,7000],
        [4.8049,551.7101,666.3636,0.8279,7000],
        [4.7742,561.3243,654.8829,0.8571,7000],        
    ]
    rrt_star_data_path2_it_7000 = [
        [4.0370,663.4968,988.5540,0.6712,7000],
        [2.6380,682.7042,944.8563,0.7225,7000],
        [4.8447,673.4701,957.3803,0.7035,7000],
        [5.4936,678.5934,956.4716,0.7095,7000],
        [5.6820,685.9920,966.5574,0.7097,7000],        
    ]
    
    # A star for two paths. Epsilon = 1, 1.5, 0.7.
    a_star_data_path1_ep_1 = [
        [6.5850, 561.4134, 680.0043, 0.8256, 36151, 1],
        [8.6259, 577.4738, 695.8476,0.8299,34913,1],
        [11.1994, 557.2540,652.6730,0.8538,84837,1],
        [6.6299, 583.5452,680.7902,0.8572,36436,1],
        [6.27275,543.0516,674.0779,0.8056,34160,1],
    ] 
    a_star_data_path2_ep_1 = [
        [15.7907,685.7966,936.3998,0.7324,65283,1],
        [12.2988, 665.3435,920.6605,0.7227,61320,1],
        [9.5889, 692.3186,942.3149,0.7347,66047,1],
        [10.1299, 680.9883,933.4707,0.7295,63730,1],
        [10.1664,674.3805,931.2819,0.7241,65394,1],
    ]
    a_star_data_path1_ep_1_5 = [
        [0.1555, 560.2607, 634.5342, 0.8829, 2504, 1.5],
        [0.1333,581.1162,637.9208,0.9110,1107,1.5],
        [0.4342,549.7217,670.4892,0.8199,9381,1.5],
        [0.1936,578.1228,664.1906,0.8704,3588,1.5],
        [1.0900,535.1205,690.1823,0.7753,20671,1.5],
    ]
    a_star_data_path2_ep_1_5 = [
        [1.0389,681.6685,951.0344,0.7168,29395,1.5],
        [1.2980,674.7814,946.8103,0.7127,28122,1.5],
        [1.1617,675.5302,940.0651,0.7186,27405,1.5],
        [1.2515,683.7287,951.1418,0.7189,31947,1.5],
        [1.3156,653.4011,942.7063,0.6931,30828,1.5],
    ]
    a_star_data_path1_ep_0_7 = [
        [1.7001,548.1761,651.1241,0.8419,47284,0.7],
        [2.3317,580.4498,636.3271,0.9122,48514,0.7],
        [1.7163,551.2722,668.8402,0.8242,48981,0.7],
        [1.7120,552.7866,668.1744,0.8273,50072,0.7],
        [2.7212,553.0145,645.9546,0.8561,46720,0.7],
    ]
    a_star_data_path2_ep_0_7 = [
        [3.6396,682.0689,936.9255,0.7280,87208,0.7],
        [3.9775,681.1615,941.5692,0.7234,87707,0.7],
        [3.8068,674.4368,939.3046,0.7180,86911,0.7],
        [3.7428,679.5366,947.0759,0.7175,88245,0.7],
        [4.0769,669.8373,938.1312,0.7140,87715,0.7],
    ]

    '''
    Scatter plot of elapsed time vs path efficiency for RRT* and A* with different parameters.
    '''
    plt.figure(figsize=(15, 6))
    # RRT* data
    plt.scatter([data[0] for data in rrt_star_data_path1_it_10000], [data[3] for data in rrt_star_data_path1_it_10000], label='RRT* Path 1 (10000 it)', color='blue')
    plt.scatter([data[0] for data in rrt_star_data_path2_it_10000], [data[3] for data in rrt_star_data_path2_it_10000], label='RRT* Path 2 (10000 it)', color='cyan')
    plt.scatter([data[0] for data in rrt_star_data_path1_it_15000], [data[3] for data in rrt_star_data_path1_it_15000], label='RRT* Path 1 (15000 it)', color='green')
    plt.scatter([data[0] for data in rrt_star_data_path2_it_15000], [data[3] for data in rrt_star_data_path2_it_15000], label='RRT* Path 2 (15000 it)', color='lime')
    plt.scatter([data[0] for data in rrt_star_data_path1_it_7000], [data[3] for data in rrt_star_data_path1_it_7000], label='RRT* Path 1 (7000 it)', color='red')
    plt.scatter([data[0] for data in rrt_star_data_path2_it_7000], [data[3] for data in rrt_star_data_path2_it_7000], label='RRT* Path 2 (7000 it)', color='magenta')
    # A* data
    plt.scatter([data[0] for data in a_star_data_path1_ep_1], [data[3] for data in a_star_data_path1_ep_1], label='A* Path 1 (epsilon=1)', color='orange')
    plt.scatter([data[0] for data in a_star_data_path2_ep_1], [data[3] for data in a_star_data_path2_ep_1], label='A* Path 2 (epsilon=1)', color='yellow')
    plt.scatter([data[0] for data in a_star_data_path1_ep_1_5], [data[3] for data in a_star_data_path1_ep_1_5], label='A* Path 1 (epsilon=1.5)', color='purple')
    plt.scatter([data[0] for data in a_star_data_path2_ep_1_5], [data[3] for data in a_star_data_path2_ep_1_5], label='A* Path 2 (epsilon=1.5)', color='violet')
    plt.scatter([data[0] for data in a_star_data_path1_ep_0_7], [data[3] for data in a_star_data_path1_ep_0_7], label='A* Path 1 (epsilon=0.7)', color='brown')
    plt.scatter([data[0] for data in a_star_data_path2_ep_0_7], [data[3] for data in a_star_data_path2_ep_0_7], label='A* Path 2 (epsilon=0.7)', color='pink')
    plt.xlabel('Elapsed Time (s)')
    plt.ylabel('Path Efficiency')
    plt.title('Elapsed Time vs Path Efficiency for RRT* and A*')
    # move legend outside of graph into the margins
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid()
    plt.savefig('elapsed_time_vs_path_efficiency_all.png', bbox_inches='tight')
    plt.show()

    # Bar chart of average path efficiency for RRT* and A* with different parameters.
    plt.figure(figsize=(15, 6))
    # RRT* data
    rrt_star_avg_efficiency_it_10000 = np.mean([data[3] for data in rrt_star_data_path1_it_10000] + [data[3] for data in rrt_star_data_path2_it_10000])
    rrt_star_avg_efficiency_it_15000 = np.mean([data[3] for data in rrt_star_data_path1_it_15000] + [data[3] for data in rrt_star_data_path2_it_15000])
    rrt_star_avg_efficiency_it_7000 = np.mean([data[3] for data in rrt_star_data_path1_it_7000] + [data[3] for data in rrt_star_data_path2_it_7000])
    plt.bar('RRT* (10000 it)', rrt_star_avg_efficiency_it_10000, color='blue')
    plt.bar('RRT* (15000 it)', rrt_star_avg_efficiency_it_15000, color='green')
    plt.bar('RRT* (7000 it)', rrt_star_avg_efficiency_it_7000, color='red')
    # A* data
    a_star_avg_efficiency_ep_1 = np.mean([data[3] for data in a_star_data_path1_ep_1] + [data[3] for data in a_star_data_path2_ep_1])
    a_star_avg_efficiency_ep_1_5 = np.mean([data[3] for data in a_star_data_path1_ep_1_5] + [data[3] for data in a_star_data_path2_ep_1_5])
    a_star_avg_efficiency_ep_0_7 = np.mean([data[3] for data in a_star_data_path1_ep_0_7] + [data[3] for data in a_star_data_path2_ep_0_7])
    plt.bar('A* (epsilon=1)', a_star_avg_efficiency_ep_1, color='orange')
    plt.bar('A* (epsilon=1.5)', a_star_avg_efficiency_ep_1_5, color='purple')
    plt.bar('A* (epsilon=0.7)', a_star_avg_efficiency_ep_0_7, color='brown')
    plt.ylabel('Average Path Efficiency')
    plt.ylim(0.75, 0.8)
    plt.title('Average Path Efficiency for RRT* and A*')
    plt.grid(axis='y')
    plt.savefig('average_path_efficiency_all.png', bbox_inches='tight')
    plt.show()

    # Bar chart of average elapsed time for RRT* and A* with different parameters.
    plt.figure(figsize=(15, 6))
    # RRT* data
    rrt_star_avg_time_it_10000 = np.mean([data[0] for data in rrt_star_data_path1_it_10000] + [data[0] for data in rrt_star_data_path2_it_10000])
    rrt_star_avg_time_it_15000 = np.mean([data[0] for data in rrt_star_data_path1_it_15000] + [data[0] for data in rrt_star_data_path2_it_15000])
    rrt_star_avg_time_it_7000 = np.mean([data[0] for data in rrt_star_data_path1_it_7000] + [data[0] for data in rrt_star_data_path2_it_7000])
    plt.bar('RRT* (10000 it)', rrt_star_avg_time_it_10000, color='blue')
    plt.bar('RRT* (15000 it)', rrt_star_avg_time_it_15000, color='green')
    plt.bar('RRT* (7000 it)', rrt_star_avg_time_it_7000, color='red')
    # A* data
    a_star_avg_time_ep_1 = np.mean([data[0] for data in a_star_data_path1_ep_1] + [data[0] for data in a_star_data_path2_ep_1])
    a_star_avg_time_ep_1_5 = np.mean([data[0] for data in a_star_data_path1_ep_1_5] + [data[0] for data in a_star_data_path2_ep_1_5])
    a_star_avg_time_ep_0_7 = np.mean([data[0] for data in a_star_data_path1_ep_0_7] + [data[0] for data in a_star_data_path2_ep_0_7])
    plt.bar('A* (epsilon=1)', a_star_avg_time_ep_1, color='orange')
    plt.bar('A* (epsilon=1.5)', a_star_avg_time_ep_1_5, color='purple')
    plt.bar('A* (epsilon=0.7)', a_star_avg_time_ep_0_7, color='brown')
    plt.ylabel('Average Elapsed Time (s)')
    plt.title('Average Elapsed Time for RRT* and A*')
    plt.grid(axis='y')
    plt.savefig('average_elapsed_time_all.png', bbox_inches='tight')
    plt.show()