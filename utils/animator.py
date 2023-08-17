from os.path import join, exists
from os import mkdir, makedirs
from config import debug, local
from IPython import display
import numpy as np
import itertools
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline
from sklearn import metrics


def use_svg_display():
    """Use the svg format to display a plot."""

    backend_inline.set_matplotlib_formats('svg')


def set_axes(axes, x_label, y_label, x_lim, y_lim, x_scale, y_scale, legend):
    """Set the axes for matplotlib.

    Defined in :numref:`sec_calculus`"""
    if x_label is not None:
        axes.set_xlabel(x_label)
    if y_label is not None:
        axes.set_ylabel(y_label)
    if x_scale is not None:
        axes.set_xscale(x_scale)
    if y_scale is not None:
        axes.set_yscale(y_scale)
    axes.set_xlim(x_lim)
    axes.set_ylim(y_lim)
    if legend:
        axes.legend(legend)
    axes.grid()


class Animator:
    """For plotting data_access in animation."""

    def __init__(self, x_label=None, y_label=None, legend=None, x_lim=None,
                 y_lim=None, x_scale='linear', y_scale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), num_rows=1, num_cols=1,
                 fig_size=(5., 3.5)):
        """init parameter for plot."""

        # Incrementally plot multiple lines
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(num_rows, num_cols, figsize=fig_size)
        if num_rows * num_cols == 1:
            self.axes = [self.axes, ]
        # Use a lambda function to capture arguments
        self.config_axes = lambda: set_axes(
            self.axes[0], x_label, y_label, x_lim, y_lim, x_scale, y_scale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # Add multiple data_access points into the figure
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        if debug is True and local is True:
            self.axes[0].cla()
            for x, y, fmt in zip(self.X, self.Y, self.fmts):
                self.axes[0].plot(x, y, fmt)
            self.config_axes()
            display.display(self.fig)
            display.clear_output(wait=True)
            plt.pause(0.01)

    def show(self):
        # Show picture
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """This function prints and plots the confusion matrix. Normalization
        can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.gcf().subplots_adjust(bottom=0.25)

    @staticmethod
    def get_confusion_matrix(cnf_matrix,
                             saving_dir: str = None,
                             model_name: str = ''):
        plt.figure()
        Animator.plot_confusion_matrix(cnf_matrix,
                                       classes=[0, 1],
                                       title='Confusion matrix, without normalization')
        plt.show()
        if saving_dir is not None:
            if not exists(saving_dir):
                makedirs(saving_dir)
            file_name = join(saving_dir, f'{model_name}_' + 'confusion')
            plt.savefig(file_name)

        plt.figure()
        Animator.plot_confusion_matrix(cnf_matrix, normalize=True, classes=[0, 1],
                                   title='Normalized confusion matrix')
        plt.show()
        if saving_dir is not None:
            if not exists(saving_dir):
                makedirs(saving_dir)
            file_name = join(saving_dir, f'{model_name}_' +  'confusion_normalized')
            plt.savefig(file_name)

    @staticmethod
    def get_metrics(effect,
                    saving_dir: str = None,
                    model_name: str = ''):
        plt.figure()
        if type(effect).__name__ == 'dict':
            ax = list(effect.keys())
            ay = list(effect.values())
            plt.ylim([0.0, 1.05])
            plt.tick_params(axis='x', labelsize=12)
            plt.bar(ax, ay)
            for a, b, i in zip(ax, ay, range(len(ax))):  # zip 函数
                plt.text(a, b + 0.01, "%.2f" % ay[i], ha='center', fontsize=12)
            plt.show()
            if saving_dir is not None:
                if not exists(saving_dir):
                    makedirs(saving_dir)
                plt.savefig(join(saving_dir, f'{model_name}_' + "metrics"))

    @staticmethod
    def get_auc(y_prob, y_true,
                saving_dir: str = None,
                model_name: str = ''):
        auc_fig = plt.figure()
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_prob, pos_label=1)
        roc_auc = metrics.auc(fpr, tpr)
        plt.figure(auc_fig.number)
        plt.plot(fpr, tpr, label=model_name + ' (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver operating characteristic (ROC)', fontsize=12)
        plt.legend(loc="lower right")
        auc_fig.show()
        if saving_dir is not None:
            if not exists(saving_dir):
                makedirs(saving_dir)
            auc_fig.savefig(join(saving_dir, f'{model_name}_' + 'auc_curves'))

    @staticmethod
    def get_auprc(y_prob, y_true,
                 saving_dir: str = None,
                 model_name: str = ''):
        prc_fig = plt.figure()
        prc_fig.set_size_inches((10, 6))
        precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_prob, pos_label=1)
        roc_auc = metrics.average_precision_score(y_true, y_prob)
        plt.figure(prc_fig.number)
        plt.plot(recall, precision, label=model_name + ' (area under precision recall curve = %0.2f)' % roc_auc)

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('precision', fontsize=12)
        plt.title('Precision Recall Curve (PRC)', fontsize=12)
        plt.legend(loc="lower right")
        prc_fig.show()
        if saving_dir is not None:
            if not exists(saving_dir):
                makedirs(saving_dir)
            prc_fig.savefig(join(saving_dir, f'{model_name}_' + 'auprc_curves'))

