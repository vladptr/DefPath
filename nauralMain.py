import torch
from torchviz import make_dot
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cv2
import numpy as np
import heapq
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

class PathFindingNetwork(nn.Module):
    def __init__(self):
        super(PathFindingNetwork, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(128 * 3 * 3, 512)
  # Фіксований розмір розгорнутого шару
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)

    def forward(self, x):
        batch_size = x.size(0)  # Отримати розмір пакету даних

        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Зміна розміру розгорнутого шару відповідно до розміру пакету даних
        x = x.view(batch_size, 128 * 3 * 3)




        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x



class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0

    def __lt__(self, other):
        return self.f < other.f

def find_yellow_and_red_circles(image_path):
    image = cv2.imread(image_path)

    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Визначення діапазону кольорів для жовтих та червоних кола
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Створення масок для жовтих та червоних кола
    yellow_mask = cv2.inRange(hsv_image, lower_yellow, upper_yellow)

    red_mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Застосування морфологічних операцій для покращення масок
    kernel = np.ones((5, 5), np.uint8)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # Знаходження контурів жовтих та червоних кола
    yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yellow_centers = []
    for contour in yellow_contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        yellow_centers.append(center)

    red_centers = []
    for contour in red_contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        red_centers.append(center)

    # Створення графа для пошуку шляху
    graph = {}
    all_centers = yellow_centers + red_centers
    for center in all_centers:
        graph[center] = []

    for i, center in enumerate(all_centers):
        distances = []
        for j, other_center in enumerate(all_centers):
            if i != j:
                distance = np.linalg.norm(np.array(center) - np.array(other_center))
                distances.append((distance, other_center))
        distances.sort()
        for distance, other_center in distances[:2]:
            graph[center].append(other_center)

    start = yellow_centers[0]
    end = yellow_centers[-1]

    # Застосування алгоритму A* для пошуку шляху
    path = astar(graph, start, end)

    # Малювання шляху на зображенні
    for i in range(len(path) - 1):
        cv2.line(image, path[i], path[i + 1], (0, 255, 0), 2)

    # Виведення зображення з намальованим шляхом
    cv2.imshow('Path', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return path


def heuristic(node, goal):
    return np.linalg.norm(np.array(node.position) - np.array(goal.position))


def astar(graph, start, end):
    open_list = []
    closed_list = []

    start_node = Node(start)
    goal_node = Node(end)

    heapq.heappush(open_list, (start_node.f, start_node))
    while open_list:
        current_node = heapq.heappop(open_list)[1]

        if current_node.position == goal_node.position:
            path = []
            while current_node:
                path.append(current_node.position)
                current_node = current_node.parent
            return path[::-1]

        closed_list.append(current_node)

        for neighbor_position in graph[current_node.position]:
            neighbor_node = Node(neighbor_position, current_node)

            if neighbor_node in closed_list:
                continue

            neighbor_node.g = current_node.g + heuristic(neighbor_node, current_node)
            neighbor_node.h = heuristic(neighbor_node, goal_node)
            neighbor_node.f = neighbor_node.g + neighbor_node.h

            if neighbor_node in open_list:
                open_node = next((node for _, node in open_list if node == neighbor_node), None)
                if neighbor_node.g < open_node.g:
                    open_list.remove((open_node.f, open_node))
                    heapq.heappush(open_list, (neighbor_node.f, neighbor_node))
            else:
                heapq.heappush(open_list, (neighbor_node.f, neighbor_node))

    return []

def train_model(image_paths, num_epochs=100, learning_rate=0.001):
    width = 25
    height = 25

    model = PathFindingNetwork()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    losses = []
    for epoch in range(num_epochs):
        total_loss = 0.0

        for image_path in image_paths:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (width, height))
            image = ToTensor()(image).unsqueeze(0)

            target_path = find_yellow_and_red_circles(image_path)

            if target_path is None:  # Add a check for None path
                continue

            optimizer.zero_grad()

            output = model(image)
            loss = criterion(output, torch.from_numpy(np.array(target_path)).float())
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        losses.append(total_loss / len(image_paths))
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(image_paths):.4f}')

    # График потерь
    plt.plot(range(1, num_epochs + 1), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()

    torch.save(model.state_dict(), 'path_to_checkpoint.pth')
    print('Training completed. Checkpoint saved.')




if __name__ == '__main__':
    image_paths = ["Q:\\SCreens\\aeros.jpg" ]
    width = 25
    height = 25
    resized_images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height))
        resized_images.append(image)

    resized_images = np.array(resized_images)
    train_model(image_paths)
    model = PathFindingNetwork()
    model.load_state_dict(torch.load('path_to_checkpoint.pth'))
    model.eval()

    input = torch.randn(1, 3, height, width)
    dot = make_dot(model(input), params=dict(list(model.named_parameters())))
    dot.render("file_name", format="png")


