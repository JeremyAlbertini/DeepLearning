import pygame
import numpy as np

OFFSET_Y  = 50
ENV_W     = 800

class Visualisation:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((1920, 1080))
        self.screen_w, self.screen_h = self.screen.get_size()
        self.nn_w = self.screen_w - ENV_W   # largeur dispo pour le réseau
        self.clock = pygame.time.Clock()
        self.list_pos_X = []
        self.list_pos_Y = []
        self.colors = []

    def init(self, nb_inputs, hidden_layers, n_outputs):
        nblayers = len(hidden_layers) + 2
        # Positions X dans la zone réseau (à droite de ENV_W)
        self.list_pos_X = [
            ENV_W + self.nn_w / (nblayers + 1) * i
            for i in range(1, nblayers + 1)
        ]
        # Positions Y pour chaque couche
        layer_sizes = [nb_inputs] + hidden_layers + [n_outputs]
        available_h = self.screen_h - OFFSET_Y * 2
        for size in layer_sizes:
            self.list_pos_Y.append([
                available_h / (size + 1) * i
                for i in range(1, size + 1)
            ])
            self.colors.append([pygame.Color(0, 0, 0) for _ in range(size)])

    def update(self, activations, state=None):
        # Couleur de la couche d'entrée
        if state is not None:
            s = np.abs(state)
            maximum = np.max(s)
            normalized = s / maximum if maximum > 0 else s
            for j, val in enumerate(normalized):
                self.colors[0][j] = pygame.Color(0, 0, int(val * 255))
        else:
            for j in range(len(self.colors[0])):
                self.colors[0][j] = pygame.Color(0, 0, 255)

        # Couleurs des couches cachées
        for i, A_layer in enumerate(activations):
            A = A_layer[:, 0]
            maximum = np.max(A)
            normalized = A / maximum if maximum > 0 else A
            for j, val in enumerate(normalized):
                self.colors[i + 1][j] = pygame.Color(0, 0, int(val * 255))

    def update_output(self, Q):
        q = Q[:, 0]
        e = np.exp(q - np.max(q))
        softmax = e / np.sum(e)
        value = q[0] > q[1]

        self.colors[-1][0] = pygame.Color(0, 0, 255 if value else 0)
        self.colors[-1][1] = pygame.Color(0, 0, 255 if not value else 0)


    def draw_env(self, frame):
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        surf = pygame.transform.scale(surf, (ENV_W, self.screen_h))
        self.screen.blit(surf, (0, 0))

    def draw(self, frame=None):
        self.screen.fill((40, 40, 40))

        if frame is not None:
            self.draw_env(frame)

        # Séparateur
        pygame.draw.line(self.screen, (80, 80, 80), (ENV_W, 0), (ENV_W, self.screen_h), 2)

        n_layers = len(self.list_pos_X)
        SEUIL = 10
        line_surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)

        for i in range(n_layers):
            if i < n_layers - 1:
                for j in range(len(self.list_pos_Y[i])):
                    val_j = self.colors[i][j].b
                    for k in range(len(self.list_pos_Y[i + 1])):
                        alpha = min(val_j, self.colors[i + 1][k].b)
                        if alpha < SEUIL:
                            continue
                        start = (self.list_pos_X[i],     self.list_pos_Y[i][j]     + OFFSET_Y)
                        end = (self.list_pos_X[i + 1], self.list_pos_Y[i + 1][k] + OFFSET_Y)
                        pygame.draw.line(line_surf, (100, 180, 255, alpha), start, end, 5)

        self.screen.blit(line_surf, (0, 0))

        for i in range(n_layers):
            for j in range(len(self.list_pos_Y[i])):
                pos = pygame.Vector2(self.list_pos_X[i], self.list_pos_Y[i][j] + OFFSET_Y)
                pygame.draw.circle(self.screen, self.colors[i][j], pos, 20)

    def render(self):
        pygame.display.flip()
        self.clock.tick(2)
