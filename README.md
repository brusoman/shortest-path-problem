# Shortest path problem


In  graph theory, the  **shortest path problem**  is the problem of finding a  path between two  vertices  (or nodes) in a  graph such that the sum of the  weights  of its constituent edges is minimized.

The problem of finding the shortest path between two intersections on a road map may be modeled as a special case of the shortest path problem in graphs, where the vertices correspond to intersections and the edges correspond to road segments, each weighted by the length of the segment.


Definition
-----------
A path in an directed graph is a sequence of vertices $P=\left(v_{1}, v_{2}, \ldots, v_{n}\right) \in V \times V \times \cdots \times V$ such that $v_{i}$ is adjacent to $v_{i+1}$  for $1 \leq i<n$. Such a path $P$ is called a path of length $n-1$ from $v_{1}$ to $v_{n}$
Let $e_{i, j}$ be the edge incident to both  $v_{i}$ and $v_{j}$

Given a real-valued weight function $f: E \rightarrow \mathbb{R}$ and an undirected (simple) graph $G$

Problem is to find the shortest path  $P=\left(v_{1}, v_{2}, \ldots, v_{n}\right)$ (where $\left.v_{1}=v \text { and } v_{n}=v^{\prime}\right)$ that over all possible $n$  minimizes the sum $$\sum_{i=1}^{n-1} f\left(e_{i, i + 1}\right)$$



Dijkstra's algorithm
-
Dijkstra's algorithm has many variants but the most common one is to find the shortest paths from the source vertex to all other vertices in the graph.

***Algorithm Steps:***

-   Set all vertices distances = $\infty$ except for the source vertex, set the source distance =  $0$.
-   Push the source vertex in a min-priority queue in the form (distance , vertex), as the comparison in the min-priority queue will be according to vertices distances.
-   Pop the vertex with the minimum distance from the priority queue (at first the popped vertex = source).
-   Update the distances of the connected vertices to the popped vertex in case of "current vertex distance + edge weight < next vertex distance", then push the vertex  
    with the new distance to the priority queue.
-   If the popped vertex is visited before, just continue without using it.
-   Apply the same algorithm again until the priority queue is empty.

***Pseudocode implementation:***
   ```
function Dijkstra(Graph, source):
       dist[source]  := 0                     // Distance from source to source is set to 0
       for each vertex v in Graph:            // Initializations
           if v ≠ source
               dist[v]  := infinity           // Unknown distance function from source to each node set to infinity
           add v to Q                         // All nodes initially in Q
      while Q is not empty:                  // The main loop
          v := vertex in Q with min dist[v]  // In the first run-through, this vertex is the source node
          remove v from Q 
          for each neighbor u of v:           // where neighbor u has not yet been removed from Q.
              alt := dist[v] + length(v, u)
              if alt < dist[u]:               // A shorter path to u has been found
                  dist[u]  := alt            // Update distance of u 
      return dist[]
  end function
```
Full code inplementation could be find  [here](https://web.archive.org/web/20131103204953/http://krasprog.ru/persons.php?page=kormyshov&blog=94)

***Time Complexity***
Time Complexity of Dijkstra's Algorithm is $O(V^2)$ but with min-priority queue it drops down to $O(V+E*log(V))$.[[Wikipedia]](https://en.wikipedia.org/wiki/Dijkstra%27s_algorithm)

***Example solution***

Problem is to find shortest path from 0 to 3 vertex

On input we have following graph:
<a href="https://imgbb.com/"><img src="https://i.ibb.co/6wd4tws/init.png" alt="init" border="0"></a>
Dijkstra algoritm defines the shortest path is 5+3+1 = 9
<a href="https://imgbb.com/"><img src="https://i.ibb.co/B2sYwb2/image.png" alt="image" border="0"></a>



 Floyd Warshall  algorithm
-
The Floyd-Warshall algorithm is an example of dynamic programming algoritms. It breaks the problem down into smaller subproblems, then combines the answers to those subproblems to solve the big, initial problem.

***Algorithm Steps:***
We initialize the solution matrix same as the input graph matrix as a first step. Then we update the solution matrix by considering all vertices as an intermediate vertex. The idea is to one by one pick all vertices and updates all shortest paths which include the picked vertex as an intermediate vertex in the shortest path. When we pick vertex number k as an intermediate vertex, we already have considered vertices ${\{0, 1, 2, .. k-1\}}$ as intermediate vertices. For every pair $(i, j)$ of the source and destination vertices respectively, there are two possible cases.

 1.  $k$ is not an intermediate vertex in shortest path from i to j. We keep the value of $dist[i][j]$ as it is.  
2.  $k$ is an intermediate vertex in shortest path from i to j. We update the value of $dist[i][j]$ as $dist[i][k] + dist[k][j]$ if $dist[i][j] > dist[i][k] + dist[k][j]$


***Pseudocode implementation:***
```
1 let dist be a |V| × |V| array of minimum distances initialized to ∞ (infinity)
2 for each edge (_u_,_v_)
3    dist[_u_][_v_] ← w(_u_,_v_)  _// the weight of the edge (_u_,_v_)_
4 for each vertex _v_
5    dist[_v_][_v_] ← 0
6 for _k_ from 1 to |V|
7    for _i_ from 1 to |V|
8       for _j_ from 1 to |V|
9          if dist[_i_][_j_] > dist[_i_][_k_] + dist[_k_][_j_] 
10             dist[_i_][_j_] ← dist[_i_][_k_] + dist[_k_][_j_]
11        end if
```
Full code inplementation could be find [here](https://www.geeksforgeeks.org/floyd-warshall-algorithm-dp-16/)

***Time Complexity***
Time Complexity of Floyd Warshall  algorithm is $O(V^3)$[[Wikipedia]](https://en.wikipedia.org/wiki/Floyd%E2%80%93Warshall_algorithm)

***Example solution***
On input we have following graph:

<a href="https://imgbb.com/"><img src="https://i.ibb.co/6wd4tws/init.png" alt="init" border="0"></a>
On output we have folloving matrix:

	 0     	 5      8      9
    INF      0      3      4
    INF    INF      0      1
    INF    INF    INF      0
From this matrix we could find that the shortest path between 0 and 3 vertex is 9


Conclusion
-
**Main Purposes:**

-   Dijkstra’s Algorithm is one example of a single-source shortest or SSSP algorithm, i.e., given a source vertex it finds shortest path from source to all other vertices.
-   Floyd Warshall Algorithm  is an example of all-pairs shortest path algorithm, meaning it computes the shortest path between all pair of nodes.

**Comparassion:**
The Floyd-Warshall algorithm is effective for calculating all shortest paths in tight graphs when there are a large number of pairs of edges between pairs of vertices. In the case of sparse graphs with edges of non-negative weight, the best choice is to use the Dijkstra algorithm for each possible node. With this choice, the difficulty is$O(|V|*|E|log(V))$ when using a binary heap, which is better than $O(|V|^3)$ in Floyd-Warshell algorithm when $|E|$ is significantly less than $|V|^3$ 


Google Collab
-
Full work code and comparison of algorithms you can find below:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg#button)](https://colab.research.google.com/drive/1a16QE95qTmuUJkfvM8tdiwN5K-BHDxmX)

Bibliography
-
- Cormen, Thomas H.; Leiserson, Charles E.; Rivest, Ronald L.; Stein, Clifford (2001). "Section 24.3: Dijkstra's algorithm". Introduction to Algorithms (Second ed.). MIT Press and McGraw–Hill. pp. 595–601.
- Cherkassky, Boris V.; Goldberg, Andrew V.; Radzik, Tomasz (1996). "Shortest paths algorithms: theory and experimental evaluation". Mathematical Programming. : 129–174.
- Abraham, Ittai; Fiat, Amos; Goldberg, Andrew V.; Werneck, Renato F. "Highway Dimension, Shortest Paths, and Provably Efficient Algorithms". ACM-SIAM Symposium on Discrete Algorithms, pages 782–793, 2010.
- Ahuja, Ravindra K.; Mehlhorn, Kurt; Orlin, James B.; Tarjan, Robert E. (April 1990). "Faster Algorithms for the Shortest Path Problem" (PDF). Journal of the ACM. 37 (2): 213–223. doi:10.1145/77600.77615.
- Thorup, Mikkel (2000). "On RAM priority Queues". SIAM Journal on Computing. 30 (1): 86–109
- Ahuja, Ravindra K.; Mehlhorn, Kurt; Orlin, James B.; Tarjan, Robert E. (April 1990). "Faster Algorithms for the Shortest Path Problem" (PDF). Journal of the ACM. 37 (2): 213–223.
- Левитин А. В. Глава 9. Жадные методы: Алгоритм Дейкстры // Алгоритмы. Введение в разработку и анализ — М.: Вильямс, 2006. — С. 189–195. — 576 с. — ISBN 978-5-8459-0987-9
- C. Анисимов. Как построить кратчайший маршрут между двумя точками
- Raman, Rajeev (1997). "Recent results on the single-source shortest paths problem". SIGACT News. 28 (2): 81–87
- Zhan, F. Benjamin; Noon, Charles E. (February 1998). "Shortest Path Algorithms: An Evaluation Using Real Road Networks". Transportation Science. 32 (1): 65–73
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE0MDAxNzc5NzgsOTkxNjExMDQwLC01NT
QyMzQ5NjksMTU5Mzk0ODA4NSw5MzUwMzczMjIsNDc5NTE2NjY1
LDczMDE4MzQxOSwtMjEzNTYyODEyMl19
-->
 BIN +453 KB docs/theory/shortest-path-problem/Shortest Path Problem.pdf 
Viewed
Binary file not shown.
 221  docs/theory/shortest-path-problem/Shortest path problem.py 
Viewed
@@ -0,0 +1,221 @@
#!/usr/bin/env python
# coding: utf-8

# In[24]:


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

#Graph Adjacency Matrix

Adj=np.asarray([[0, 156, 0, 0, 246, 0, 184, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 462, 0, 0, 171, 0, 157, 0, 363], 
[156, 0, 323, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 323, 0, 151, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 151, 0, 0, 545, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[246, 0, 0, 0, 0, 174, 0, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 545, 174, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[184, 0, 0, 0, 0, 0, 0, 83, 224, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 192, 0, 0, 0], 
[0, 0, 0, 0, 100, 0, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 224, 0, 0, 209, 0, 0, 0, 0, 217, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 209, 0, 116, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 116, 0, 180, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 180, 0, 157, 251, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 157, 0, 342, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 251, 342, 0, 111, 208, 0, 0, 0, 0, 0, 382, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 217, 0, 0, 0, 0, 111, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 208, 0, 0, 335, 462, 0, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 335, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[462, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 462, 0, 0, 212, 0, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 212, 0, 135, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 135, 0, 174, 0, 0, 0, 0], 
[171, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 0, 0, 0, 0, 0], 
[0, 0, 0, 0, 0, 0, 192, 0, 0, 0, 0, 0, 0, 382, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
[157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 171, 0], 
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 171, 0, 0], 
[363, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])



#Plot the directed graph
G = nx.DiGraph()

N = Adj.shape[0]
for i in range(N):
    G.add_node(i)

for i in range(N):
    for j in range(N):
        if Adj[i,j] > 0:
            G.add_edges_from([(i, j)], weight=Adj[i,j])

print("Graph plotting:")

pos=nx.spring_layout(G) 
edge_labels=dict([((u,v,),d['weight'])
                 for u,v,d in G.edges(data=True)]) 
nx.draw_networkx(G,pos,edge_labels=edge_labels, node_size = 1000, node_color = 'y') 
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.axis('off')
plt.show()

#Dijkstra algorithm
distance = np.zeros(N) 
visited = np.ones(N) 
origin = 0
goal = 4

visited[origin] = 0

pred = np.zeros(N)
pred[origin] = origin

for j in range(N):
  if Adj[origin,j] == 0 and origin != j: 
    distance[j] = 10e10
    pred[j] = -1
  else:
    distance[j] = Adj[origin,j]
    pred[j] = origin

array = []
array.append(0)
counter = 1

while(np.sum(visited) > 0): 
  temp = np.copy(distance) 
  temp[visited == 0] = 10e10
  vmin = np.argmin(temp)

  visited[vmin] = 0
  for j in range(N):
    counter+=1
    if Adj[vmin,j] > 0 and distance[j] > distance[vmin] + Adj[vmin,j]: 
      distance[j] = distance[vmin]+Adj[vmin,j]
      array.append(counter)
      pred[j] = vmin

#print("counter:", array)
pred = pred.astype(int) #Minimum distance path from origin node to the others
#print("Pred")
#print(pred)
#Plot path

dist_list = []
prev_list = []

dist_list.append(distance[goal])
prev_list.append(goal)

previous = pred[goal]
path = [(previous, goal),(goal, previous)]
print("The minimum distance path from "+str(origin)+" to "+str(goal)+" is: "+str(goal)+" <-- "+str(previous), end="")



while(previous != origin):
  path.append((previous, pred[previous]))
  path.append((pred[previous], previous))
  dist_list.append(distance[previous])
  prev_list.append(previous)
  previous = pred[previous]
  print(" <-- "+str(previous), end="")
dist_list.append(distance[previous])

#print(dist_list)
#print(prev_list)

edge_colors = ['black' if not edge in path else 'red' for edge in G.edges()]

pos=nx.spring_layout(G)
edge_labels=dict([((u,v,),d['weight'])
                 for u,v,d in G.edges(data=True)])
nx.draw_networkx(G,pos,edge_labels=edge_labels, node_size = 1000, node_color = 'y', edge_color=edge_colors)
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.axis('off')
plt.show()


dist_list.reverse()
dist_list = [dist_list[-1] - x for x in dist_list]

plt.plot(array[:len(dist_list)], dist_list, label='Dijkstra')
plt.legend()
plt.xlabel('Iterations') 

plt.ylabel('Distance')
plt.show()


# In[25]:


from matplotlib import pyplot as plt
import numpy as np
''' Part of Cosmos by OpenGenus Foundation '''
INF = 1000000000

def floyd_warshall(vertex, adjacency_matrix,a,b):


    # calculating all pair shortest path
    counter = 0
    arr_value = []
    arr_counter = []
    prev_value= 0



    for k in range(0, vertex):
        for i in range(0, vertex):
            for j in range(0, vertex):
                # relax the distance from i to j by allowing vertex k as intermediate vertex
                # consider which one is better, going through vertex k or the previous value
                counter += 1
                adjacency_matrix[i][j] = min(adjacency_matrix[i][j], adjacency_matrix[i][k] + adjacency_matrix[k][j])
                if(adjacency_matrix[a][b] != prev_value ):
                    prev_value = adjacency_matrix[i][j]
                    if(prev_value != INF):
                        arr_counter.append(counter)
                        arr_value.append(prev_value)
                        #print(prev_value)
                        #print(counter)





    return arr_counter, arr_value, adjacency_matrix[a][b]


for i in range(len(Adj)):
    for j in range(len(Adj)):
        if(Adj[i][j] == 0 and i != j):
            Adj[i][j] = INF

x, y, max_value = floyd_warshall(len(Adj), Adj, 0, 4)

y = [max_value - i for i in y]

plt.plot(x, y, label='Floyd–Warshall')
plt.legend()
plt.xlabel('Iterations') 
plt.ylabel('Distance')


# In[26]:


plt.plot(x, y, label='Floyd–Warshall')
plt.plot(array[:len(dist_list)], dist_list, label='Dijkstra')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Distance')


# In[ ]:
