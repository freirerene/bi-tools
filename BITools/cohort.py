import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _cohort_period(df: pd.DataFrame) -> pd.DataFrame:
    df['cohort_period'] = np.arange(len(df)) + 1
    return df


class CreateCohort:

    def __init__(self, df: pd.DataFrame, customer_id: str, created_at: str,
                 calculating='customer_id', function='nunique',
                 save_as='cohort', show=False):

        self.df = df
        self.customer_id = customer_id
        self.created_at = created_at

        self.df[self.created_at] = pd.to_datetime(self.df[self.created_at])
        self.df['order_period'] = self.df[self.created_at].dt.strftime('%Y-%m')

        self.df.set_index(customer_id, inplace=True)

        self.df['cohort'] = self.df.groupby(level=0)[self.created_at].min().apply(lambda x: x.strftime('%Y-%m'))

        self.df.reset_index(inplace=True)

        self.grouped = self.df.groupby(['cohort', 'order_period'])

        if function == 'nunique':
            self.cohorts = self.grouped.agg({calculating: pd.Series.nunique})
        elif function == 'sum':
            self.cohorts = self.grouped.agg({calculating: pd.Series.sum})
        elif function == 'mean':
            self.cohorts = self.grouped.agg({calculating: pd.Series.mean})

        self.cohorts = self.cohorts.groupby(level=0).apply(_cohort_period)

        self.cohorts.reset_index(inplace=True)
        self.cohorts.set_index(['cohort', 'cohort_period'], inplace=True)

        cohort_group_size = self.cohorts[calculating].groupby(level=0).first()

        self.user_retention = self.cohorts[calculating].unstack(0).divide(cohort_group_size, axis=1)

        self.user_retention.T.to_csv(save_as + '.csv')

        self.fig = plt.figure(figsize=(15, 12))
        ax = self.fig.add_subplot(1, 1, 1)

        sns.heatmap(self.user_retention.T, annot=True, cmap="YlGnBu", fmt='.2g', cbar=False, ax=ax)
        plt.yticks(rotation='360')
        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        plt.rcParams['figure.dpi'] = 150
        plt.rcParams['savefig.dpi'] = 150

        plt.savefig(save_as + '.png')
        if show:
            plt.show()
        else:
            plt.close()
