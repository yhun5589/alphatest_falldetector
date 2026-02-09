import os
import sys
from pathlib import Path
import winreg as reg  # Windows Registry

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.metrics import dp

FILE_NAME = "userids.txt"
Window.size = (400, 600)

def add_to_startup(exe_path: str, name="BRServer"):
    """
    Add exe_path to Windows startup if not already present.
    """
    exe_path = str(Path(exe_path).resolve())
    key = r"Software\Microsoft\Windows\CurrentVersion\Run"
    try:
        registry_key = reg.OpenKey(reg.HKEY_CURRENT_USER, key, 0, reg.KEY_READ)
        try:
            reg.QueryValueEx(registry_key, name)
            reg.CloseKey(registry_key)
            print(f"{name} already in startup.")
            return
        except FileNotFoundError:
            reg.CloseKey(registry_key)
            # Add it
            registry_key = reg.OpenKey(reg.HKEY_CURRENT_USER, key, 0, reg.KEY_WRITE)
            reg.SetValueEx(registry_key, name, 0, reg.REG_SZ, exe_path)
            reg.CloseKey(registry_key)
            print(f"Added {name} to startup.")
    except Exception as e:
        print(f"Failed to modify registry: {e}")

class IDManager(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', padding=15, spacing=15, **kwargs)

        # Scrollable area for input boxes
        self.scroll = ScrollView(size_hint=(1, 1))
        self.grid = GridLayout(cols=1, spacing=10, size_hint_y=None, padding=5)
        self.grid.bind(minimum_height=self.grid.setter('height'))
        self.scroll.add_widget(self.grid)
        self.add_widget(self.scroll)

        # Buttons area
        btn_layout = BoxLayout(size_hint_y=None, height=50, spacing=10)
        self.add_btn = Button(
            text="➕ Add ID",
            size_hint_x=0.5,
            background_normal='',
            background_color=(0.0, 0.6, 1, 1),
            color=(1, 1, 1, 1),
            bold=True
        )
        self.add_btn.bind(on_press=self.add_input)

        self.set_btn = Button(
            text="💾 Save IDs",
            size_hint_x=0.5,
            background_normal='',
            background_color=(0.0, 0.8, 0.2, 1),
            color=(1, 1, 1, 1),
            bold=True
        )
        self.set_btn.bind(on_press=self.save_ids)

        btn_layout.add_widget(self.add_btn)
        btn_layout.add_widget(self.set_btn)
        self.add_widget(btn_layout)

        # Load saved IDs
        self.load_ids()

    def add_input(self, instance=None, text=""):
        ti = TextInput(
            text=text,
            hint_text="Enter User ID...",
            size_hint_y=None,
            height=dp(45),
            multiline=False,
            padding=(12, 12),
            background_normal='',
            background_active='',
            background_color=(1, 1, 1, 1),
            foreground_color=(0, 0, 0, 1),
            cursor_color=(0, 0, 0, 1)
        )
        self.grid.add_widget(ti)
        ti.focus = True

    def save_ids(self, instance=None):
        # Save IDs to text file
        ids = [child.text.strip() for child in self.grid.children if isinstance(child, TextInput) and child.text.strip()]
        with open(FILE_NAME, "w") as f:
            f.write("\n".join(reversed(ids)))
        print(f"Saved {len(ids)} IDs to {FILE_NAME}.")

        # Determine exe folder
        exe_dir = Path(sys.executable).parent if getattr(sys, 'frozen', False) else Path(__file__).parent
        brserver_path = exe_dir / "brserver.exe"
        print(f"Looking for brserver.exe at: {brserver_path}")

        if brserver_path.exists():
            add_to_startup(brserver_path)
        else:
            print("brserver.exe not found in the exe folder!")

    def load_ids(self):
        if os.path.exists(FILE_NAME):
            with open(FILE_NAME, "r") as f:
                for line in f:
                    self.add_input(text=line.strip())
        if not self.grid.children:
            self.add_input()

class UserIDApp(App):
    def build(self):
        self.title = "User ID Manager"
        return IDManager()

if __name__ == "__main__":
    UserIDApp().run()
