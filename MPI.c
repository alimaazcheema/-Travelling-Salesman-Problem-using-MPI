#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <time.h>
#include <mpi.h>

#define N 8 // Number of cities

// Function to generate random distances between cities
void generateRandomDistances(int graph[N][N]) {
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                graph[i][j] = 0; // Distance from a city to itself is 0
            } else {
                // Generate random distance between 10 and 100
                graph[i][j] = rand() % 91 + 10;
            }
        }
    }
}

// Function to print the graph (for debugging purposes)
void printGraph(int graph[N][N]) {
    printf("Randomly generated distances:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d\t", graph[i][j]);
        }
        printf("\n");
    }
}

// Function to calculate total distance of a path
int totalDistance(int path[], int n, int graph[N][N]) {
    int total = 0;
    for (int i = 0; i < n - 1; i++) {
        total += graph[path[i]][path[i + 1]];
    }
    total += graph[path[n - 1]][path[0]]; // Return to starting city
    return total;
}

// Function to perform parallel branch-and-bound for TSP
void parallelTSP(int rank, int size) {
    int citiesPerProcess = N / size;

    // Determine cities assigned to this process
    int startCity = rank * citiesPerProcess;
    int endCity = (rank == size - 1) ? N : startCity + citiesPerProcess;

    // Create graph and generate random distances
    int graph[N][N];
    generateRandomDistances(graph);

    // Uncomment the following line to print the randomly generated distances
    // printGraph(graph);

    int bestPath[N]; // Stores the best path found so far
    int minDistance = INT_MAX; // Stores the minimum distance found so far

    // Initialize the initial path and distance
    int initialPath[N];
    for (int i = 0; i < N; i++) {
        initialPath[i] = i;
    }

    // Initialize the stack for branch-and-bound
    int stack[N + 1];
    int stackSize = 0;
    stack[stackSize++] = 0; // Start with the first city

    // Perform branch-and-bound search
    while (stackSize > 0) {
        // Pop the top node from the stack
        int city = stack[--stackSize];

        // Check if we've visited all cities
        if (stackSize == N) {
            int currentDistance = totalDistance(stack, N, graph);
            if (currentDistance < minDistance) {
                minDistance = currentDistance;
                for (int i = 0; i < N; i++) {
                    bestPath[i] = stack[i];
                }
            }
            continue;
        }

        // Branch to the next level
        for (int i = endCity - 1; i >= startCity; i--) {
            int newPath[N];
            for (int j = 0; j < city; j++) {
                newPath[j] = stack[j];
            }
            newPath[city] = i;
            stack[stackSize++] = i;
        }
    }

    // Gather results from all processes to process 0
    if (rank == 0) {
        int globalBestPath[N];
        MPI_Gather(bestPath, citiesPerProcess, MPI_INT, globalBestPath, citiesPerProcess, MPI_INT, 0, MPI_COMM_WORLD);

        int globalMinDistance;
        MPI_Reduce(&minDistance, &globalMinDistance, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);

        // Output global best path and distance
        printf("Global Best Path: ");
        for (int i = 0; i < N; i++) {
            printf("%d ", globalBestPath[i]);
        }
        printf("\nGlobal Minimum Distance: %d\n", globalMinDistance);
    } else {
        MPI_Gather(bestPath, citiesPerProcess, MPI_INT, NULL, 0, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Reduce(&minDistance, NULL, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size > N) {
        if (rank == 0) {
            printf("Number of processes cannot exceed number of cities.\n");
        }
        MPI_Finalize();
        return 0;
    }

    if (N % size != 0) {
        if (rank == 0) {
            printf("Number of cities must be divisible by number of processes.\n");
        }
        MPI_Finalize();
        return 0;
    }

    parallelTSP(rank, size);

    MPI_Finalize();
    return 0;
}
