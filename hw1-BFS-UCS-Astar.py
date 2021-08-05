import copy
from collections import deque
import heapq
import math
import time
start = time.time()

input = open('input.txt','r')

algorithm = input.readline()
algorithm = algorithm.rstrip('\n')

temp = input.readline()
dimension = [int(s) for s in temp.split() if s.isdigit()]

temp = input.readline()
entrance = [int(s) for s in temp.split() if s.isdigit()]
entrance = tuple(entrance)

temp = input.readline()
exit = [int(s) for s in temp.split() if s.isdigit()]
exit = tuple(exit)

point_num = int(input.readline())

point_loc_list = []
point_act_list = {}
for i in range(0,point_num):
	temp = input.readline()
	a_point = [int(s) for s in temp.split() if s.isdigit()]
	x = a_point.pop(0)
	y = a_point.pop(0)
	z = a_point.pop(0)
	a_point_loc = (x,y,z)
	a_point_act = tuple(a_point)
	point_loc_list.append(a_point_loc)
	point_act_list[a_point_loc] = a_point_act


def code_to_action(loc, action): #checked
	if action == 1:
		return (loc[0]+1,loc[1],loc[2])
	elif action == 2:
		return (loc[0]-1,loc[1],loc[2])
	elif action == 3:
		return (loc[0],loc[1]+1,loc[2])
	elif action == 4:
		return (loc[0],loc[1]-1,loc[2])
	elif action == 5:
		return (loc[0],loc[1],loc[2]+1)
	elif action == 6:
		return (loc[0],loc[1],loc[2]-1)
	elif action == 7:
		return (loc[0]+1,loc[1]+1,loc[2])
	elif action == 8:
		return (loc[0]+1,loc[1]-1,loc[2])
	elif action == 9:
		return (loc[0]-1,loc[1]+1,loc[2])
	elif action == 10:
		return (loc[0]-1,loc[1]-1,loc[2])
	elif action == 11:
		return (loc[0]+1,loc[1],loc[2]+1)
	elif action == 12:
		return (loc[0]+1,loc[1],loc[2]-1)
	elif action == 13:
		return (loc[0]-1,loc[1],loc[2]+1)
	elif action == 14:
		return (loc[0]-1,loc[1],loc[2]-1)
	elif action == 15:
		return (loc[0],loc[1]+1,loc[2]+1)
	elif action == 16:
		return (loc[0],loc[1]+1,loc[2]-1)
	elif action == 17:
		return (loc[0],loc[1]-1,loc[2]+1)
	elif action == 18:
		return (loc[0],loc[1]-1,loc[2]-1)
	else :
		return -1

#build graph
graph = {}
for point in point_loc_list:
	for action in point_act_list[point]:
		if point in graph:
			graph[point].append(code_to_action(point, action))
		else:
			graph[point] = [code_to_action(point, action)]

print(algorithm)
#for key, value in graph.items():
    #print(key, ' : ', value)
#print(dimension)
#print(entrance)#only location
#print(exit)    #only location
#print(point_num)
#print(point_loc_list)
#print(point_act_list)


def BFS(graph, entrance, exit): #可以改point act list {坐标：动作}

	queue = deque([entrance])

	visited_node = {entrance:0}
	path = {entrance:0}

	optimal_path = []
	while queue:
		point = queue.popleft()
		step = visited_node[point]

		for item in graph[point]: #all neighbors
			if item not in visited_node:
				visited_node[item] = step + 1
				queue.append(item)
				path[item] = point
			elif visited_node[item] > step + 1:
				visited_node[item] = step + 1
				queue.append(item)
				path[item] = point

	if exit in visited_node:
		key = exit
		while key in path:
			optimal_path.insert(0,key)
			key = path[key]
	else:
		optimal_path = []

	return optimal_path

def BFS_generate_output(path):
	output = open('output.txt','w')
	if len(path) == 0:
		output.write("FAIL")
		return 0

	step_num = len(path)
	total_cost = step_num-1

	output.write(str(total_cost))
	output.write("\n")
	output.write(str(step_num))
	output.write("\n")

	for i in range(0,step_num):
		for j in range(0,3):
			output.write(str(path[i][j]))
			output.write(" ")
		if i == 0:
			output.write("0")
		else:
			output.write("1")
		if i != step_num - 1:
			output.write("\n")
	output.close()

def distance(point1, point2, mode):
	dis = math.sqrt( (point2[0]-point1[0])**2 + (point2[1]-point1[1])**2 + (point2[2]-point1[2])**2 )
	if mode == "near":
		if dis > 1:
			return 14
		else:
			return 10
	if mode == "far":
		dis = int(dis*10)
		return dis

def UCS(graph, entrance, exit):
	queue = [ (0,entrance) ]

	visited_node = {entrance:0}
	path = {entrance:0}

	optimal_path = []
	optimal_cost = 0
	while queue:
		q_head = heapq.heappop(queue) #tuple (cost, [path])
		path_cost = q_head[0]
		point = q_head[1]#last point of current path


		for item in graph[point]: #all neighbors
			dis = distance(point, item, "near")
			new_cost = path_cost + dis

			if item not in visited_node:
				visited_node[item] = new_cost
				heapq.heappush(queue, (new_cost, item) )
				path[item] = point
			elif visited_node[item] > new_cost:
				visited_node[item] = new_cost
				heapq.heappush(queue, (new_cost, item) )
				path[item] = point

	if exit in visited_node:
		optimal_cost = visited_node[exit]
		key = exit
		while key in path:
			optimal_path.insert(0,key)
			key = path[key]
	else:
		optimal_path = []
		optimal_cost = 0

	return optimal_path, optimal_cost

def A_star(graph, entrance, exit):
	queue = [ (0,entrance) ]

	visited_node = {entrance:0} #past cost
	est_v_node = {entrance:0} #estimated cost
	path = {entrance:0}

	optimal_path = []
	optimal_cost = 0
	while queue:
		q_head = heapq.heappop(queue) #tuple (cost, [path])
		point = q_head[1]#last point of current path
		path_cost = visited_node[point]


		for item in graph[point]: #all neighbors
			dis = distance(point, item, "near")
			new_cost = path_cost + dis
			fut_est = distance(item, exit, "far")
			est_cost = new_cost + fut_est

			if item not in est_v_node:
				visited_node[item] = new_cost
				est_v_node[item] = est_cost
				heapq.heappush(queue, (est_cost, item) )
				path[item] = point

			elif est_v_node[item] > est_cost:
				visited_node[item] = new_cost
				est_v_node[item] = est_cost
				heapq.heappush(queue, (est_cost, item) )
				path[item] = point

	if exit in visited_node:
		optimal_cost = visited_node[exit]
		key = exit
		while key in path:
			optimal_path.insert(0,key)
			key = path[key]
	else:
		optimal_path = []
		optiaml_cost = 0

	return optimal_path, optimal_cost

def UCS_A_output(path, total_cost):
	output = open('output.txt','w')
	if len(path) == 0:
		output.write("FAIL")
		return 0

	step_num = len(path)

	output.write(str(total_cost))
	output.write("\n")
	output.write(str(step_num))
	output.write("\n")

	for i in range(0,step_num):
		for j in range(0,3):
			output.write(str(path[i][j]))
			output.write(" ")
		if i == 0:
			output.write("0")
		else:
			cost = distance(path[i], path[i-1], "near")
			output.write(str(cost))
		if i != step_num - 1:
			output.write("\n")
	output.close()


if algorithm == "BFS":
	result = BFS(graph, entrance, exit)
	BFS_generate_output(result)

elif algorithm == "UCS":
	path, total_cost = UCS(graph, entrance, exit)
	UCS_A_output(path, total_cost)

elif algorithm == "A*":
	path, total_cost = A_star(graph, entrance, exit)
	UCS_A_output(path, total_cost)








input.close()
end = time.time()
cost = end - start
print("total time:")
print(cost)
