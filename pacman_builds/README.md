# Pacman Builds
## grid_data_obs
### Observation
| Observation Name                  | Position in Observation |
|-----------------------------------|-------------------------|
| Pacman Coord (x,y,z)              | [0, 1, 2]               |
| Pacman State (is/isn't attacking) | [3]                     |
| Input Direction (from Agent)      | [4]                     |
| Movement Direction                | [5, 6]                  |
| Ghost 1 Coord                     | [7, 8]                  |
| Ghost 1 State                     | [9]                     |
| Ghost 2 Coord                     | [10, 11]                |
| Ghost 2 State                     | [12]                    |
| Ghost 3 Coord                     | [13, 14]                |
| Ghost 3 State                     | [15]                    |
| Ghost 4 Coord                     | [16, 17]                |
| Ghost 4 State                     | [18]                    |
| Power Pellet States               | [19, 20, 21, 22]        |
| Score                             | [23]                    |
| Number of lives                   | [24]                    |
| Number of Pellets Remaining       | [25]                    |
| Number of Power Pellets Remaining | [26]                    |
| Fruit 1 State                     | [27]                    |
| Fruit 2 State                     | [28]                    |
| Closest Pellet                    | [29]                    |
| Boolean Grid                      | [30, ...]               |


### Reward
| Reward for               |   Reward value    |
|--------------------------|:-----------------:|
| Step                     |  -0.25 per frame  |
| Collecting pellet        |        +10        |
| Collecting Power Pellets |        +50        |
| Eating Ghost             |       +200        |
| Clearing level           |       +1000       |
| Eaten By Ghost           |       -500        |
| Loosing all lives        |       -1000       |
| Cherry                   |       +100        |