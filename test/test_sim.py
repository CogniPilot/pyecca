from pyecca2.att_est import launch, plot
import os



def test_sim():
    fig_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')
    data = launch.run_montecarlo_sim(n=8)
    plot.plot(data, fig_dir)
    return data


if __name__ == "__main__":
    data = test_sim()
