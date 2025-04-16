import matplotlib.pyplot as plt


def plot_the_spline(xi, yi, moving_car_name):
    # Plot the points as a line
    plt.figure(figsize=(10, 6))
    # plt.cla()
    ax = plt.gca()
    ax.invert_yaxis()
    plt.plot(xi, yi, label='Line Plot')

    # Add labels and title
    plt.xlabel('Y-axis')
    plt.ylabel('X-axis')
    plt.title(f'Bezier Curve For {moving_car_name}')

    # Show legend if needed
    plt.legend()

    # Display the plot
    plt.show()


def plot_vehicle_relative_path(points, moving_car_name):
    plt.figure(figsize=(10, 6))
    plt.cla()
    ax = plt.gca()
    #ax.invert_xaxis()

    # Extract x and y coordinates
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    # x_coords = x_coords[::8]
    # y_coords = y_coords[::8]

    # Plotting with reversed x-axis
    plt.plot(y_coords, x_coords, marker='o', linestyle='-')
    plt.title(f'{moving_car_name} Actual Path')
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')

    # # Reverse the x-axis
    # plt.xlim(max(x_coords), min(x_coords))

    plt.grid(True)
    plt.show()


def combine_plot(xi, yi, points, moving_car_name):
    plt.figure(figsize=(10, 6))
    plt.cla()
    ax = plt.gca()

    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]

    print('-' * 75)
    # -1 to convert to airsim for the plot
    print(f"route start at: {xi[0], -1 * yi[0]}")
    print(f"car start: {points[0][0], points[0][1]}")
    print('-' * 75)
    # -1 to convert to airsim for the plot
    plt.plot(-1 * yi, xi, color='red', label='bezier')
    # plt.plot(y_coords, x_coords, marker='o', linestyle='-',color='green',label='actual')
    plt.plot(y_coords, x_coords, color='green', label='actual')

    plt.title(f'{moving_car_name} Actual Path')
    plt.xlabel('Y Coordinate')
    plt.ylabel('X Coordinate')

    plt.legend()
    plt.show()


def plot_vehicle_object_path(list_of_points):
    """
    Plots multiple vehicle paths from a list of lists of points.

    Args:
        list_of_points: A list of lists, where each inner list represents a vehicle's path
                        as a series of [x, y] coordinates.
    """

    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    ax.invert_yaxis()

    axis1 = [12, 18, 24, 30, 36]
    axis2 = [12, 12, 12, 12, 12]
    axis3 = [0, 0, 0, 0, 0]
    plt.plot(axis1, axis2, linestyle='dashed', color='black', linewidth=2)
    plt.plot([-x for x in axis1], axis2, linestyle='dashed', color='black', linewidth=2)
    plt.plot(axis1, [-y for y in axis2], linestyle='dashed', color='black', linewidth=2)
    plt.plot([-x for x in axis1], [-y for y in axis2], linestyle='dashed', color='black', linewidth=2)

    plt.plot(axis2, axis1, linestyle='dashed', color='black', linewidth=2)
    plt.plot([-y for y in axis2], axis1, linestyle='dashed', color='black', linewidth=2)
    plt.plot(axis2, [-x for x in axis1], linestyle='dashed', color='black', linewidth=2)
    plt.plot([-y for y in axis2], [-x for x in axis1], linestyle='dashed', color='black', linewidth=2)

    plt.plot(axis1, axis3, linestyle='dashed', color='dimgray', linewidth=2)
    plt.plot([-x for x in axis1], axis3, linestyle='dashed', color='dimgray', linewidth=2)
    plt.plot(axis3, [-y for y in axis1], linestyle='dashed', color='dimgray', linewidth=2)
    plt.plot(axis3, axis1, linestyle='dashed', color='dimgray', linewidth=2)
    # Set the tick marks for both axes at intervals of 6, 12, 18, etc.
    ax.set_xticks(range(-30, 31, 6))
    ax.set_yticks(range(-30, 31, 6))

    # Define a color list for different vehicles (feel free to customize colors)
    colors = ['b', 'g', 'r', 'y']
    # Loop through each list of points and plot them in a different color
    for vehicle_points, car_name in list_of_points:
        x_coords = [point[0] for point in vehicle_points]
        y_coords = [point[1] for point in vehicle_points]
        i = int(car_name[-1])
        if i == 1 :
            x_offset = 4
            y_offset = 3
        elif i == 2:
            x_offset = 0
            y_offset = 2
        elif i == 3 :
            x_offset = 5
            y_offset = -4
        else : # i == 4:
            x_offset = 5
            y_offset = -4

        plt.plot(x_coords, y_coords, marker='o', linestyle='-', label=f" {car_name}", color=colors[i % len(colors)])
        plt.text(x_coords[0] + x_offset, y_coords[0] + y_offset, f'{car_name} start', fontsize=11,
                 va='top', ha='right', fontweight='bold')
        plt.scatter(x_coords[0], y_coords[0], color='black', s=100, zorder=5)

    plt.title('Vehicle Paths - Airsim Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()  # Add legend to identify vehicle paths
    plt.show()