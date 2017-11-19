from IPython.display import display
from ipywidgets import interactive, Layout
import ipywidgets as widgets
from collections import OrderedDict

import os
import matplotlib.pyplot as plt
from evgena.dataset.loaders import _load_idx_mnist, load_idx_emnist


# TODO enlarge example slider
class DatasetViewer:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.label_map = None
        self.X = [None]
        self.y = [None]
        self.is_inverted = False

        self.fig, self.ax = plt.subplots()

        dataset_picker_wi = widgets.ToggleButtons(
            options=OrderedDict([
                ('balanced', lambda: load_idx_emnist(os.path.normpath('../datasets/emnist_balanced'))),
                ('letters', lambda: load_idx_emnist(os.path.normpath('../datasets/emnist_letters'))),
                ('digits', lambda: load_idx_emnist(os.path.normpath('../datasets/emnist_digits'))),
                # ('mnist (emnist)', lambda: load_idx_emnist(os.path.normpath('../evgena/dataset/emnist_mnist'))),
                # ('mnist (original)', lambda: _load_idx_mnist(os.path.normpath('../evgena/dataset/mnist')))
            ]),
            description='Dataset:',
            disabled=False,
            button_style='',
            tooltips=[
                'Loads EMNIST balanced dataset',
                'Loads EMNIST letters dataset',
                'Loads EMNIST digits dataset',
                # 'Loads EMNIST mnist dataset',
                # 'Loads original MNIST dataset',
            ]
        )

        suite_picker_wi = widgets.ToggleButtons(
            options=OrderedDict([
                ('train', lambda: (self.X_train, self.y_train)),
                ('test', lambda: (self.X_test, self.y_test))
                # TODO consider adding validation (last 20% of train)
            ]),
            description='Suite:',
            disabled=False,
            button_style='',
            tooltips=[
                'Load images from train suite',
                'Load images from test suite',
            ]
        )

        example_picker_wi = widgets.IntSlider(
            value=0,
            description='Index:',
            continuous_update=False,
            min=0,
            max=len(self.X) - 1,
            step=1,
            disabled=False
        )

        invert_toggle_wi = widgets.ToggleButton(
            value=True,
            description='Invert colours',
            disabled=False,
            button_style='',
            tooltip='Invert example to human friendly colours',
            icon='check'
        )

        self.dataset_picker = interactive(
            self.load_dataset,
            fnc=dataset_picker_wi
        )

        self.suite_picker = interactive(
            self.load_suite,
            fnc=suite_picker_wi
        )

        self.example_picker = interactive(
            self.show_example,
            index=example_picker_wi
        )

        self.invert_toggle = interactive(
            self.invert_colours,
            is_inverted=invert_toggle_wi
        )

        self.ui_box = widgets.VBox(
            children=[
                self.dataset_picker,
                self.suite_picker,
                widgets.HBox(
                    children=[
                        self.example_picker,
                        self.invert_toggle
                    ]
                )
            ]
        )

        display(self.ui_box)

    def load_dataset(self, fnc):
        ((self.X_train, self.y_train), (self.X_test, self.y_test), self.label_map) = fnc()
        self.suite_picker.update()

    def load_suite(self, fnc):
        self.X, self.y = fnc()
        self.example_picker.children[0].max = len(self.X) - 1
        self.example_picker.update()

    def show_example(self, index):
        title = 'Label: {l}; raw label: {r!r}'.format(l=', '.join(self.label_map[self.y[index]]), r=self.y[index])
        self.fig.set_label(title)
        self.fig.canvas.set_window_title(title)
        self.ax.imshow(self.X[index] if not self.is_inverted else 255 - self.X[index], cmap='gray')
        plt.show()

    def invert_colours(self, is_inverted):
        self.is_inverted = is_inverted
        self.example_picker.update()
