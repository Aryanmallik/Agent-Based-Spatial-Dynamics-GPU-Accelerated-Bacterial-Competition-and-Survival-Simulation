import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.widgets import Button

kernel_code = """
__kernel void five_species_competition(__global int* species, __global int* positions, __global float* directions, 
                                  int num_bacteria, int width, int height, __global float* wiggle,
                                  __global float* growth_rates, __global int* competition_matrix,
                                  int current_time, int virus_time) {
    int i = get_global_id(0);
    if (i >= num_bacteria || species[i] == -1) return;

    float angle = directions[i];
    int dx = round(cos(angle));
    int dy = round(sin(angle));

    int new_x = (positions[2 * i] + dx + width) % width;
    int new_y = (positions[2 * i + 1] + dy + height) % height;

    positions[2 * i] = new_x;
    positions[2 * i + 1] = new_y;

    directions[i] += wiggle[i];

    int neighbors[4][2] = {
        {new_x, (new_y + 1) % height},
        {new_x, (new_y - 1 + height) % height},
        {(new_x + 1) % width, new_y},
        {(new_x - 1 + width) % width, new_y}
    };

    for (int j = 0; j < num_bacteria; j++) {
        if (j == i || species[j] == -1) continue;
        int other_x = positions[2 * j];
        int other_y = positions[2 * j + 1];
        int other_species = species[j];

        for (int k = 0; k < 4; k++) {
            if (other_x == neighbors[k][0] && other_y == neighbors[k][1]) {
                int my_species = species[i];
                
                if (my_species == 2 && other_species == -2) {
                    if (current_time - virus_time > 200) {
                        float transform_prob = 0.5f;
                        float rand_val = (float)((i+1) * (j+1)) / (float)(num_bacteria * num_bacteria);
                        
                        if (rand_val < transform_prob) {
                            species[i] = 5;
                            species[j] = -1;
                        }
                    }
                }
                else if (my_species == 5 && other_species == -2) {
                    float convert_prob = 0.8f;
                    float rand_val = (float)((i+1) * (j+1)) / (float)(num_bacteria * num_bacteria);
                    
                    if (rand_val < convert_prob) {
                        species[j] = 5;
                    }
                }
                else if (other_species >= 0) {
                    if (my_species <= 5 && other_species <= 5) {
                        int idx = my_species * 6 + other_species;
                        
                        if (competition_matrix[idx] == 1) {
                            float rand_val = (float)((i+1) * (j+1)) / (float)(num_bacteria * num_bacteria);
                            float competition_prob = growth_rates[my_species] * 10.0f;
                            
                            if (rand_val < competition_prob) {
                                species[j] = species[i];
                            }
                        }
                    }
                }
            }
        }
    }
    
    if (species[i] >= 0 && species[i] <= 5 && growth_rates[species[i]] > 0.0) {
        float reproduction_prob = growth_rates[species[i]];
        float rand_val = (float)(i * 123) / (float)(num_bacteria * 100);
        
        if (rand_val < reproduction_prob) {
        }
    }
}

__kernel void virus_attack(__global int* species, __global int* positions, int num_bacteria, int width, int height) {
    int i = get_global_id(0);
    if (i >= num_bacteria || species[i] == -1) return;
    
    int x = positions[2 * i];
    int y = positions[2 * i + 1];
    
    if (species[i] == -2) {
        float death_roll = (float)((i+1) * 12345) / (float)(num_bacteria * 10000);
        if (death_roll < 0.2f) {
            species[i] = -1;
            return;
        }
        
        for (int j = 0; j < num_bacteria; j++) {
            if (j == i || species[j] == -1 || species[j] == -2) continue;
            
            int other_x = positions[2 * j];
            int other_y = positions[2 * j + 1];
            
            bool is_adjacent = (
                (abs(x - other_x) <= 1 || abs(x - other_x) >= width - 1) && 
                (abs(y - other_y) <= 1 || abs(y - other_y) >= height - 1)
            );
            
            if (is_adjacent && species[j] != 2 && species[j] != 5) {
                float infection_roll = (float)((i+1) * (j+1)) / (float)(num_bacteria * num_bacteria);
                if (infection_roll < 0.3f) {
                    species[j] = -2;
                }
            }
        }
    }
}
"""

platforms = cl.get_platforms()
gpu_devices = [device for platform in platforms for device in platform.get_devices(device_type=cl.device_type.GPU)]
if gpu_devices:
    device = gpu_devices[-1]
else:
    device = cl.get_platforms()[0].get_devices(device_type=cl.device_type.CPU)[0]
    
ctx = cl.Context([device])
queue = cl.CommandQueue(ctx)
program = cl.Program(ctx, kernel_code).build()

width, height = 500, 500
num_bacteria = 10000

species_names = {
    0: "Toxin Producers (Red)",
    1: "Fast Growers (Green)",
    2: "Resistant (Blue)",
    3: "Scavengers (Purple)",
    4: "Symbiotic (Yellow)",
    5: "Super Bacteria (Cyan)",
    -2: "Infected (White)"
}

species_count = num_bacteria // 5
species = np.array([0] * species_count + [1] * species_count + [2] * species_count + 
                  [3] * species_count + [4] * species_count, dtype=np.int32)
np.random.shuffle(species)

positions = np.random.choice(width * height, size=num_bacteria, replace=False)
positions = np.column_stack((positions % width, positions // width))

directions = np.random.uniform(0, 2 * np.pi, size=num_bacteria).astype(np.float32)
wiggle = np.random.uniform(-0.2, 0.2, size=num_bacteria).astype(np.float32)

growth_rates = np.array([0.03, 0.05, 0.02, 0.04, 0.025, 0.06], dtype=np.float32)

competition_matrix = np.zeros((6, 6), dtype=np.int32)

competition_matrix[0, 1] = 1
competition_matrix[0, 4] = 1

competition_matrix[1, 2] = 1
competition_matrix[1, 3] = 1

competition_matrix[2, 0] = 1
competition_matrix[2, 4] = 1

competition_matrix[3, 0] = 1
competition_matrix[3, 2] = 1

competition_matrix[4, 1] = 1
competition_matrix[4, 3] = 1

competition_matrix[5, 0] = 1
competition_matrix[5, 1] = 1
competition_matrix[5, 2] = 1
competition_matrix[5, 3] = 1
competition_matrix[5, 4] = 1

competition_matrix_flat = competition_matrix.flatten()

species_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=species)
positions_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=positions)
directions_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=directions)
wiggle_buf = cl.Buffer(ctx, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=wiggle)
growth_rates_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=growth_rates)
competition_matrix_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=competition_matrix_flat)

colors = {
    0: [0.9, 0.1, 0.1],
    1: [0.1, 0.9, 0.1],
    2: [0.1, 0.1, 0.9],
    3: [0.8, 0.2, 0.8],
    4: [0.9, 0.9, 0.1],
    5: [0.0, 0.8, 0.8],
    -1: [0, 0, 0],
    -2: [1, 1, 1]
}

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('black')
ax.set_facecolor('black')

plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.15, wspace=0, hspace=0)
ax.margins(0)
ax.set_xticks([])
ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

img = ax.imshow(np.zeros((height, width, 3)), animated=True, interpolation='nearest')
title = ax.set_title("Five Species Bacterial Ecosystem", color="white", fontsize=16)

legend_patches = []
for i in range(6):
    patch = mpatches.Patch(color=colors[i], label=species_names[i])
    legend_patches.append(patch)
patch = mpatches.Patch(color=colors[-2], label=species_names[-2])
legend_patches.append(patch)
legend = ax.legend(handles=legend_patches, loc='lower center', bbox_to_anchor=(0.5, -0.1),
                   ncol=4, facecolor='black', edgecolor='white', labelcolor='white')

button_ax = fig.add_axes([0.4, 0.02, 0.2, 0.05])
virus_button = Button(button_ax, 'Toggle Virus', color='red', hovercolor='darkred')
virus_button.label.set_color('white')

text = ax.text(0.01, 0.97, "", fontsize=10, color="white", ha='left', va='top',
               transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.7, pad=5))

virus_text = ax.text(0.99, 0.02, "No virus active", fontsize=10, color="white", ha='right', va='bottom',
                    transform=ax.transAxes, bbox=dict(facecolor='black', alpha=0.7, pad=5))

network_ax = fig.add_axes([0.75, 0.75, 0.2, 0.2], facecolor='black')
network_ax.set_xticks([])
network_ax.set_yticks([])
for spine in network_ax.spines.values():
    spine.set_visible(False)

positions_network = {
    0: [0.5, 0.9],
    1: [0.9, 0.5],
    2: [0.5, 0.1],
    3: [0.1, 0.5],
    4: [0.5, 0.5],
    5: [0.2, 0.2],
}

for i in range(6):
    network_ax.add_patch(plt.Circle(positions_network[i], 0.1, color=colors[i]))
    network_ax.text(positions_network[i][0], positions_network[i][1], str(i),
                   ha='center', va='center', color='white', fontsize=8)

for i in range(6):
    for j in range(6):
        if i < 5 and j < 5 and competition_matrix[i, j] == 1:
            dx = positions_network[j][0] - positions_network[i][0]
            dy = positions_network[j][1] - positions_network[i][1]
            arrow_length = np.sqrt(dx**2 + dy**2)
            dx = dx * 0.7 / arrow_length
            dy = dy * 0.7 / arrow_length
            start_x = positions_network[i][0] + dx * 0.1/0.7
            start_y = positions_network[i][1] + dy * 0.1/0.7
            end_x = positions_network[j][0] - dx * 0.1/0.7
            end_y = positions_network[j][1] - dy * 0.1/0.7
            network_ax.arrow(start_x, start_y, end_x-start_x, end_y-start_y, 
                           head_width=0.05, head_length=0.05, fc=colors[i], ec=colors[i])
        elif i == 5 and competition_matrix[i, j] == 1:
            dx = positions_network[j][0] - positions_network[i][0]
            dy = positions_network[j][1] - positions_network[i][1]
            arrow_length = np.sqrt(dx**2 + dy**2)
            dx = dx * 0.7 / arrow_length
            dy = dy * 0.7 / arrow_length
            start_x = positions_network[i][0] + dx * 0.1/0.7
            start_y = positions_network[i][1] + dy * 0.1/0.7
            end_x = positions_network[j][0] - dx * 0.1/0.7
            end_y = positions_network[j][1] - dy * 0.1/0.7
            network_ax.arrow(start_x, start_y, end_x-start_x, end_y-start_y, 
                          head_width=0.05, head_length=0.05, fc=colors[i], ec=colors[i])

network_ax.set_xlim(0, 1)
network_ax.set_ylim(0, 1)
network_ax.set_title("Competition Network", color='white', fontsize=10)

population_graph = fig.add_axes([0.05, 0.75, 0.2, 0.2])
population_graph.set_facecolor('black')
for spine in population_graph.spines.values():
    spine.set_color('white')
population_graph.tick_params(colors='white')
population_graph.set_title('Population Dynamics', color='white', fontsize=10)

lines = {}
for i in range(6):
    lines[i] = population_graph.plot([], [], color=colors[i], linewidth=1.5, label=f"Species {i}")[0]

population_graph.set_xlim(0, 300)
population_graph.set_ylim(0, num_bacteria)
population_graph.set_ylabel('Count', color='white', fontsize=8)
population_graph.set_xlabel('Time', color='white', fontsize=8)

population_history = {i: [] for i in range(6)}
timesteps = []
current_timestep = 0
virus_active = False
virus_start_time = 0

def trigger_virus(event):
    global virus_active, virus_start_time
    virus_active = not virus_active
    
    if virus_active:
        virus_start_time = current_timestep
        
        cl.enqueue_copy(queue, species, species_buf)
        initial_infections = np.random.choice(num_bacteria, size=20, replace=False)
        for idx in initial_infections:
            if species[idx] != 2 and species[idx] != 5:
                species[idx] = -2
        cl.enqueue_copy(queue, species_buf, species)
        
        virus_text.set_text("VIRUS ACTIVE!")
        virus_text.set_color("red")
        print("Virus released! Blue (Resistant) bacteria are immune.")
    else:
        virus_text.set_text("No virus active")
        virus_text.set_color("white")
        print("Virus deactivated.")

virus_button.on_clicked(trigger_virus)

def update(frame):
    global current_timestep
    current_timestep += 1
    
    cl.enqueue_copy(queue, wiggle_buf, np.random.uniform(-0.2, 0.2, size=num_bacteria).astype(np.float32))
    
    program.five_species_competition(
        queue, (num_bacteria,), None,
        species_buf, positions_buf, directions_buf,
        np.int32(num_bacteria), np.int32(width), np.int32(height),
        wiggle_buf, growth_rates_buf, competition_matrix_buf,
        np.int32(current_timestep), np.int32(virus_start_time)
    )
    
    if virus_active:
        program.virus_attack(
            queue, (num_bacteria,), None,
            species_buf, positions_buf, np.int32(num_bacteria), 
            np.int32(width), np.int32(height)
        )
    
    cl.enqueue_copy(queue, species, species_buf)
    cl.enqueue_copy(queue, positions, positions_buf)
    cl.enqueue_copy(queue, directions, directions_buf)
    
    time_to_cyan = max(0, 200 - (current_timestep - virus_start_time)) if virus_active else 0
    seconds_to_cyan = time_to_cyan // 50
    
    species_counts = [np.sum(species == i) for i in range(6)]
    infected_count = np.sum(species == -2)
    
    for i in range(6):
        population_history[i].append(species_counts[i])
    timesteps.append(current_timestep)
    
    extinct_species = [i for i in range(6) if species_counts[i] == 0 and i in species_names]
    if extinct_species:
        extinct_names = [species_names[i] for i in extinct_species]
        print(f"Extinction event! {', '.join(extinct_names)} went extinct.")
    
    surviving_species = [i for i in range(6) if species_counts[i] > 0]
    if len(surviving_species) == 1:
        winner = surviving_species[0]
        print(f"Ecosystem dominated by {species_names[winner]}")
    
    grid = np.zeros((height, width, 3))
    for i in range(num_bacteria):
        if species[i] != -1:
            x, y = positions[i]
            grid[y, x] = colors[species[i]]
    img.set_array(grid)
    
    counts_text = "\n".join([f"{species_names[i]}: {species_counts[i]}" for i in range(6) if i in species_names])
    counts_text += f"\n{species_names[-2]}: {infected_count}"
    
    if virus_active and seconds_to_cyan > 0:
        counts_text += f"\n\nCyan Evolution: {seconds_to_cyan} counts remaining"
    elif virus_active and time_to_cyan == 0 and species_counts[5] == 0:
        counts_text += f"\n\nCyan Evolution: READY!"
    
    text.set_text(counts_text)
    
    if frame % 5 == 0:
        if len(timesteps) > 300:
            start_idx = len(timesteps) - 300
        else:
            start_idx = 0
            
        for i in range(6):
            lines[i].set_data(timesteps[start_idx:], population_history[i][start_idx:])
        
        population_graph.set_xlim(max(0, current_timestep - 300), max(300, current_timestep))
    
    return_elements = [img, text, title, legend, virus_text]
    for i in range(6):
        return_elements.append(lines[i])
    
    return return_elements

ani = animation.FuncAnimation(fig, update, frames=None, interval=20, blit=False)
plt.show()
