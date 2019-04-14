import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import seaborn as sns; sns.set()
import math

#only for bianry classification
def see_all_2(train, test,features,target,bins,dump_root="../see_all/"):
    nBinLevel = len(bins);  assert nBinLevel>1
    values, counts = np.unique(train.target, return_counts=True)
    for f_name in features:
        n = 0
        print("see_all_2@...".format(f_name), end="")
        fig, ax = plt.subplots(nBinLevel, 2, figsize=(20, 10))
        #a = train[f_name].loc[train.target == 0]
        fig.suptitle("\"{}\" V={} N={} 0={} 1={}".format(f_name,values, counts,"Blue","Red"))
        for bin in bins:
            bin = bin if bin >0 else None
            sns.distplot(train[f_name].loc[train.target == 0],ax=ax[n,0], color="Blue", bins=bin,norm_hist=True)
            sns.distplot(train.loc[train.target == 1, f_name],ax=ax[n,0], color="Red", bins=bin, norm_hist=True)
            sns.distplot(test.loc[:, f_name],ax=ax[n,1], color="Mediumseagreen", bins=bin, norm_hist=True)
            #ax[0].set_xlabel("")
            #ax[1].set_xlabel("")
            n=n+1
        #plt.show(block=True)
        plt.savefig("{}_[{}]_.jpg".format(dump_root, f_name))
        plt.clf();        plt.cla();        plt.close()

def plot_binary_dist(train,test,feature_names,bins=None):
    n_top = max(2,len(feature_names))       #1-1D array of subplots.
    fig, ax = plt.subplots(n_top, 2, figsize=(10, 5 * n_top))        #, figsize=(10, 5 * n_top)
    n=0
    for f_name in feature_names:
        a = train[f_name].loc[train.target == 0]
        sns.distplot(train[f_name].loc[train.target == 0], ax=ax[n, 0], color="Blue", bins=bins,norm_hist=True)
        sns.distplot(train.loc[train.target == 1, f_name], ax=ax[n, 0], color="Red", bins=bins, norm_hist=True)
        sns.distplot(test.loc[:, f_name], ax=ax[n, 1], color="Mediumseagreen", bins=bins, norm_hist=True)
        ax[n, 0].set_title("Train {}".format(f_name))
        ax[n, 1].set_title("Test {}".format(f_name))
        ax[n, 0].set_xlabel("")
        ax[n, 1].set_xlabel("")
        n=n+1
    plt.show(block=True)

def ann(row,col_A,col_B,axis=None):
    ind = row[0]
    r = row[1]
    info = "{}:{:.2g}".format(r[col_A],r[col_B])     #ind
    info = "{:.2g}".format(r[col_B])
    plt.gca().annotate(info, xy=(r[col_A], r[col_B]), xytext=(2,2) , textcoords ="offset points" )

def plot_join_distri(df,listDict):
    no,nFig = 0,len(listDict)
    listG=[]
    for dict in listDict:
        x,y,title=dict['x'], dict['y'], dict['title']
        sns.set(style="darkgrid", color_codes=True)
        #marginal = dict(bins=15, rug=True)
        marginal = {'bins':150, 'rug':True}
        g = sns.jointplot(x, y, data=df, kind="reg",size=10,marginal_kws=marginal)        #kind=
        if False:
            g = g.plot_joint(plt.scatter, color="m", edgecolor="white")
            _ = g.ax_marg_x.hist(x, color="b", alpha=.6)
            _ = g.ax_marg_y.hist(y, color="r", alpha=.6,orientation="horizontal")
        head = df.sort_values(by=[x], ascending=[False]).head(5)
        #tail = tips.sort_values(by=['resid'], ascending=[False]).tail(5)
        for row in head.iterrows():
            ann(row,x,y)
        plt.title(title)
        path = "{}_{}.png".format(title, no)
        g.savefig(path);        listG.append(path)
        #plt.close()     #必须close 不然plt.show()会重复
        no = no + 1

    if False:        # subplots migration
        fig = plt.figure(figsize=(2, 2))    #    plt.figure(figsize=(12, 8))
        no=0
        for path in listG:
            no = no + 1
            img=mpimg.imread(path)
            fig.add_subplot(2, 2, no)
            plt.imshow(img)

    plt.show()
    print("listG={}".format(len(listG)))

#https://stackoverflow.com/questions/43010462/annotate-outliers-on-seaborn-jointplot
#很多问题，会丢失坐标轴
def plot_join_distri_0(df,listDict):
    no,nFig = 0,len(listDict)
    nRow=int(math.sqrt(nFig))
    nCol=(int)(math.ceil(nFig*1.0/nRow))
    fig, axs = plt.subplots(nRow,nCol)
    for dict in listDict:
        x,y,title=dict['x'], dict['y'], dict['title']
        row,col=(int)(no/nCol),(int)(no%nCol)
        axis = axs[row,col]
        g = sns.jointplot(x, y, data=df, kind="reg",size=7, ax=axis)
        head = df.sort_values(by=[x], ascending=[False]).head(5)
        #tail = tips.sort_values(by=['resid'], ascending=[False]).tail(5)
        for row in head.iterrows():
            ann(row,x,y,axis)
        no=no+1
        plt.title(title)
        plt.close()  # 必须close 不然plt.show()会重复

    plt.show()
    print("listG={}".format(0))
# https://stackoverflow.com/questions/35042255/how-to-plot-multiple-seaborn-jointplot-in-subplot
class SeabornFig2Grid():

    def __init__(self, seaborngrid, fig,  subplot_spec):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)

        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())

if __name__ == "__main__":
    iris = sns.load_dataset("iris")
    tips = sns.load_dataset("tips")

    # An lmplot
    g0 = sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,palette=dict(Yes="g", No="m"))
    # A PairGrid
    g1 = sns.PairGrid(iris, hue="species")
    g1.map(plt.scatter, s=5)
    # A FacetGrid
    g2 = sns.FacetGrid(tips, col="time",  hue="smoker")
    g2.map(plt.scatter, "total_bill", "tip", edgecolor="w")
    # A JointGrid
    g3 = sns.jointplot("sepal_width", "petal_length", data=iris,kind="kde", space=0, color="g")
    fig = plt.figure(figsize=(13,8))
    gs = gridspec.GridSpec(2, 2)
    mg0 = SeabornFig2Grid.SeabornFig2Grid(g0, fig, gs[0])
    mg1 = SeabornFig2Grid.SeabornFig2Grid(g1, fig, gs[1])
    mg2 = SeabornFig2Grid.SeabornFig2Grid(g2, fig, gs[3])
    mg3 = SeabornFig2Grid.SeabornFig2Grid(g3, fig, gs[2])
    gs.tight_layout(fig)
    #gs.update(top=0.7)
    plt.show()