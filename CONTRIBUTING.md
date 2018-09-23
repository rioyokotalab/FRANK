# Logs and conclusions

Logs and conclusions for meetings taking place:

** 8 Aug 2018 **

Meeting between Sameer and Peter.

Make a simple MPI code for the current class structure.

The code will have the following features:
* Make a basic code that is split across 4 processes.
* Don't make it block cyclic. Instead, keep it cyclic and simply split 4 blocks across 4 processes.
* If there is further split down any level, it stays on the same processor.
* We do not handle recursively split multi-process blocks at this point of time.
* Run this through Yokota sensei and set timeline.

** 9 Aug 2018  **

Meeting between Sameer and Yokota-sensei.

We improved on yesterday's discussion between Sameer and Peter. Yesterday
we had thought of two ways of implementing the splitting: one kind of matrix
block is distributed and the other one is not. The one which is not distributed
is too small to qualify for distribution and therefore we need to split it accoross
many processes, we can just compute it on a single process and distribute the results.

However, Yokota-sensei pointed out that this can lead to imbalance since only a corner
process will be kept busy and all the others will need to wait for the communication from it.

He suggested a new way of looking at it:
* Only distribute the full rank matrices until it no longer makes sense to distribute
them further.
* Never compute a full LU decomposition of a subdivided block only on a single process.
Always keep the decomposition distributed.
* Since it does not seem efficient to distribute the low rank block accross processes,
duplicate the low rank block and the reduction across all the processes. The communication
from the full rank blocks (for the L and U parts) will then basically be a broadcast of the
computed blocks to all the processes so they do the (duplicated) computation by themselves.

Here's a photo of the whiteboard:

![9 aug 2018 whiteboard](images/9_8_18.jpeg**

** 21 Sept 2018 **

Meeting between Sameer and Yokota-sensei.

We discussed how the current hicma code can be modified to make it MPI-compatible. We came
up with a novel scheme for distributing hierarchically recursive blocks over multiple processes.
The scheme is inspired by SLATE and will allow us to change the granularity of the
computation at runtime. We can also change how exactly we want the data distribution of a
block and not be restricted to exclusively using block-cyclic distribution of all blocks.

Following is a summary of the new approach:

Copy SLATE design methodologies.
Dissociate creation of matrix object from memory allocation.
Have some way of determinining which object is present on which node.

A 'tile' is SLATE is actually the underlying matrix block (Dense/LR/HMat) in hicma.

On object creation:
+ Specify the "N", "NB" and number of processes over which the matrix will be distributed
over.
+ This will be executed by each process and that will create a 'map' in each process.
+ There are two cases in this case:
  - The first time the matrix is created and there is no data.
  - There is already data in the matrix and it needs to be split further.

+ First case of no data:
  - Simply create the map in each process so that each process will come to know 
    which block it owns and does not own.
+ Second case of with data:
  - In this case, the process that creates the split has two options:
    - Keep the block within itself.
    - Broadcast it to all other processes.
  - If it keeps it within itself it does not need to do anything special since all
    computation will be handled on one process itself.
  - If it decides to broadcast, it will send a broadcast to all process with its data
    and communicate the new map to them.

Pros of mapping approach:
+ Since we don't know how well the algorithm can load balance, we can dynamically
change the process mapping depending on the kind of process block that we encounter
simply by changing the process mapping.
+ Thus we make no assumptions about the splitting of the data when working with
the matrix for the first time.
+ Changing the process distribution of the block is basically a matter of changing
the way the map is stored and then distributing the data accordingly.

# Contribution guidelines

Document all the classes/functions you write. Follow this guide:
