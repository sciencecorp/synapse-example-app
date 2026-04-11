#!/usr/bin/env python3
"""
BCI Bit Rate Challenge Game
Reads joystick_out tap from the Synapse app and runs a 60-second
4-direction selection task. Score = B = log2(N-1) * max(Sc-Si, 0) / t
"""

import argparse
import math
import random
import sys
import time

import numpy as np
import pygame

try:
    from synapse.api.datatype_pb2 import Tensor
    from synapse.client.taps import Tap
    HAS_SYNAPSE = True
except ImportError:
    HAS_SYNAPSE = False
    print("WARNING: synapse not installed - running in KEYBOARD demo mode")

N              = 4
DIRECTIONS     = ['UP', 'DOWN', 'LEFT', 'RIGHT']
JOY_THRESHOLD  = 0.30
DWELL_FRAMES   = 3          # 3 frames at 30 Hz ~= 100 ms dwell
COOLDOWN_FRAMES = 8         # ~= 270 ms between selections at 30 Hz
EVAL_DURATION  = 60

W, H = 1280, 800
BG      = (12,  12,  22)
WHITE   = (245, 245, 255)
GRAY    = (90,  90,  115)
DIM     = (40,  40,  60)
ACCENT  = (60,  145, 255)
GREEN   = (50,  220, 80)
RED     = (220, 55,  55)
YELLOW  = (255, 210, 50)
ORANGE  = (255, 145, 30)


def decode_direction(x, y, threshold=JOY_THRESHOLD):
    if abs(x) < threshold and abs(y) < threshold:
        return None
    if abs(x) >= abs(y):
        return 'RIGHT' if x > 0 else 'LEFT'
    return 'UP' if y > 0 else 'DOWN'


def bit_rate(N, Sc, Si, t):
    if t <= 0 or Sc <= Si:
        return 0.0
    return math.log2(N - 1) * (Sc - Si) / t


def draw_arrow(surface, direction, cx, cy, size, color):
    s = size
    if direction == 'UP':
        pts = [(cx, cy - s), (cx - s*0.65, cy + s*0.5), (cx + s*0.65, cy + s*0.5)]
    elif direction == 'DOWN':
        pts = [(cx, cy + s), (cx - s*0.65, cy - s*0.5), (cx + s*0.65, cy - s*0.5)]
    elif direction == 'LEFT':
        pts = [(cx - s, cy), (cx + s*0.5, cy - s*0.65), (cx + s*0.5, cy + s*0.65)]
    elif direction == 'RIGHT':
        pts = [(cx + s, cy), (cx - s*0.5, cy - s*0.65), (cx - s*0.5, cy + s*0.65)]
    else:
        return
    pygame.draw.polygon(surface, color, pts)


class JoystickReader:
    def __init__(self, device_ip, tap_name='joystick_out'):
        self.x = 0.0
        self.y = 0.0
        self.connected = False
        self._tap = None
        if not HAS_SYNAPSE or not device_ip:
            return
        try:
            self._tap = Tap(device_ip)
            self.connected = self._tap.connect(tap_name)
            print(f"[tap] {'Connected' if self.connected else 'FAILED'}: '{tap_name}' @ {device_ip}")
        except Exception as e:
            print(f"[tap] Exception: {e}")

    def poll(self):
        if not self._tap:
            return
        try:
            raw = self._tap.read()
            if raw is None:
                return
            tensor = Tensor()
            tensor.ParseFromString(raw)
            arr = np.frombuffer(tensor.data, dtype=np.float32)
            if arr.size >= 2:
                self.x = float(arr[0])
                self.y = float(arr[1])
        except Exception:
            pass

    def disconnect(self):
        if self._tap:
            try:
                self._tap.disconnect()
            except Exception:
                pass


class BCIGame:
    def __init__(self, device_ip, tap_name='joystick_out'):
        pygame.init()
        self.screen = pygame.display.set_mode((W, H))
        pygame.display.set_caption('BCI Bit Rate Challenge')
        self.f_xl = pygame.font.Font(None, 200)
        self.f_lg = pygame.font.Font(None, 90)
        self.f_md = pygame.font.Font(None, 52)
        self.f_sm = pygame.font.Font(None, 34)
        self.f_xs = pygame.font.Font(None, 26)

        self.reader = JoystickReader(device_ip, tap_name)
        self.state = 'WAITING'
        self.target = None
        self.Sc = self.Si = 0
        self.start_time = None
        self.elapsed = 0.0
        self.bps = 0.0
        self.dwell_dir = None
        self.dwell_count = 0
        self.cooldown = 0
        self.flash_color = None
        self.flash_until = 0
        self.clock = pygame.time.Clock()
        self._demo_mode = not HAS_SYNAPSE or not device_ip
        self._kb_x = 0.0
        self._kb_y = 0.0

    def start_game(self):
        self.Sc = self.Si = 0
        self.elapsed = 0.0
        self.bps = 0.0
        self.dwell_dir = None
        self.dwell_count = 0
        self.cooldown = 0
        self.flash_color = None
        self.start_time = time.time()
        self.state = 'PLAYING'
        self._pick_target()

    def _pick_target(self):
        self.target = random.choice(DIRECTIONS)
        self.dwell_dir = None
        self.dwell_count = 0

    def _register_selection(self, direction):
        correct = (direction == self.target)
        self.Sc += 1 if correct else 0
        self.Si += 0 if correct else 1
        self.flash_color = GREEN if correct else RED
        self.flash_until = pygame.time.get_ticks() + 180
        self.cooldown = COOLDOWN_FRAMES
        self._pick_target()

    def update(self):
        if self.state != 'PLAYING':
            return
        self.elapsed = time.time() - self.start_time
        if self.elapsed >= EVAL_DURATION:
            self.state = 'DONE'
            return
        self.reader.poll()
        if self._demo_mode:
            self.reader.x = self._kb_x
            self.reader.y = self._kb_y
        if self.cooldown > 0:
            self.cooldown -= 1
            return
        decoded = decode_direction(self.reader.x, self.reader.y)
        if decoded is None:
            self.dwell_dir = None
            self.dwell_count = 0
        elif decoded == self.dwell_dir:
            self.dwell_count += 1
            if self.dwell_count >= DWELL_FRAMES:
                self._register_selection(decoded)
                self.dwell_count = 0
        else:
            self.dwell_dir = decoded
            self.dwell_count = 1
        if self.elapsed > 0:
            self.bps = bit_rate(N, self.Sc, self.Si, self.elapsed)

    def _blit_centered(self, surf, y):
        self.screen.blit(surf, (W // 2 - surf.get_width() // 2, y))

    def draw(self):
        if self.state == 'WAITING':
            self._draw_waiting()
        elif self.state == 'PLAYING':
            self._draw_playing()
        elif self.state == 'DONE':
            self._draw_done()
        pygame.display.flip()

    def _draw_waiting(self):
        self.screen.fill(BG)
        self._blit_centered(self.f_lg.render('BCI Bit Rate Challenge', True, WHITE), 80)
        lines = [
            ('Tilt left joystick to match the arrow on screen', GRAY),
            ('Hold direction for ~100 ms to register a selection', GRAY),
            ('', None),
            (f'Alphabet size  N = {N}   |   60-second evaluation', WHITE),
            ('', None),
            ('[ SPACE ]  to begin', YELLOW),
        ]
        y = 240
        for text, color in lines:
            if color is None:
                y += 20
                continue
            self._blit_centered(self.f_sm.render(text, True, color), y)
            y += 48
        if self._demo_mode:
            st, sc = 'DEMO MODE - use arrow keys', ORANGE
        elif self.reader.connected:
            st, sc = 'Tap connected', GREEN
        else:
            st, sc = 'Tap NOT connected - check IP and that app is started', RED
        self._blit_centered(self.f_sm.render(st, True, sc), H - 50)

    def _draw_playing(self):
        now = pygame.time.get_ticks()
        if self.flash_color and now < self.flash_until:
            r, g, b = self.flash_color
            self.screen.fill((r // 6, g // 6, b // 6))
        else:
            self.screen.fill(BG)
            self.flash_color = None
        cx, cy = W // 2, H // 2 + 10
        if self.target:
            arrow_color = (self.flash_color if (self.flash_color and now < self.flash_until)
                           else WHITE)
            draw_arrow(self.screen, self.target, cx, cy, 130, arrow_color)
        BAR_W, BAR_H = 320, 18
        bx, by = cx - BAR_W // 2, cy + 185
        pygame.draw.rect(self.screen, DIM, (bx - 2, by - 2, BAR_W + 4, BAR_H + 4))
        if self.dwell_dir and self.cooldown == 0:
            fill = int(BAR_W * self.dwell_count / DWELL_FRAMES)
            pygame.draw.rect(self.screen,
                             GREEN if self.dwell_dir == self.target else RED,
                             (bx, by, fill, BAR_H))
        remaining = max(0.0, EVAL_DURATION - self.elapsed)
        tc = RED if remaining < 10 else YELLOW if remaining < 20 else WHITE
        self._blit_centered(self.f_lg.render(f'{remaining:.1f}s', True, tc), 18)
        self._blit_centered(self.f_md.render(f'{self.bps:.2f}  bps', True, ACCENT), 110)
        self._blit_centered(self.f_sm.render(f'N={N}    Sc={self.Sc}    Si={self.Si}', True, GRAY), 162)
        jt = self.f_xs.render(f'joy  x={self.reader.x:+.2f}  y={self.reader.y:+.2f}', True, DIM)
        self.screen.blit(jt, (16, H - 28))
        dec = decode_direction(self.reader.x, self.reader.y)
        dt = self.f_xs.render(f'decoded: {dec or "-"}', True, ACCENT if dec else DIM)
        self.screen.blit(dt, (W - dt.get_width() - 16, H - 28))

    def _draw_done(self):
        self.screen.fill(BG)
        self._blit_centered(self.f_md.render('Session Complete', True, YELLOW), 60)
        final_bps = bit_rate(N, self.Sc, self.Si, self.elapsed)
        self._blit_centered(self.f_xl.render(f'{final_bps:.2f}', True, WHITE), 110)
        self._blit_centered(self.f_md.render('bits per second', True, GRAY), 310)
        acc = 100 * self.Sc / max(1, self.Sc + self.Si)
        y = 390
        for text, color in [
            (f'N  =  {N}',                      WHITE),
            (f'Sc =  {self.Sc}   (correct)',     GREEN),
            (f'Si =  {self.Si}   (incorrect)',   RED),
            (f't  =  {self.elapsed:.1f} s',       GRAY),
            (f'Accuracy  {acc:.1f} %',           GRAY),
        ]:
            self._blit_centered(self.f_sm.render(text, True, color), y)
            y += 46
        self._blit_centered(self.f_xs.render('[ R ] restart    [ Q ] quit', True, DIM), H - 40)

    def run(self):
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    k = event.key
                    if k == pygame.K_q:
                        running = False
                    elif k == pygame.K_r and self.state == 'DONE':
                        self.state = 'WAITING'
                    elif k == pygame.K_SPACE and self.state == 'WAITING':
                        self.start_game()
                    elif self._demo_mode and self.state == 'PLAYING':
                        if k == pygame.K_UP:    self._kb_x, self._kb_y = 0.0,  0.8
                        elif k == pygame.K_DOWN:  self._kb_x, self._kb_y = 0.0, -0.8
                        elif k == pygame.K_LEFT:  self._kb_x, self._kb_y = -0.8, 0.0
                        elif k == pygame.K_RIGHT: self._kb_x, self._kb_y =  0.8, 0.0
                elif event.type == pygame.KEYUP and self._demo_mode:
                    self._kb_x, self._kb_y = 0.0, 0.0
            self.update()
            self.draw()
            self.clock.tick(60)
        self.reader.disconnect()
        pygame.quit()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device-ip', default='')
    parser.add_argument('--tap-name', default='joystick_out')
    args = parser.parse_args()
    BCIGame(device_ip=args.device_ip, tap_name=args.tap_name).run()

if __name__ == '__main__':
    main()
