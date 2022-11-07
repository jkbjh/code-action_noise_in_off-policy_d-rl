import nox
import sys

def common_install(session):
    session.install("--upgrade", "pip")
    session.install("-r", "requirements.built")
    # session.install('.')


@nox.session(python="3.8")
def smoke_test(session):
    print(sys.version)
    common_install(session)
    with session.cd("source"):
        session.run("python", "-m",
                    "nb_exploration_metrics.colored_action_noise_deepdeterministic",
                    "--algorithm", "detsac", "--envname", "HopperPyBulletEnv-v0",
                    "--eval-interval-factor", "2.0", "--evaluate-ntimes", "100",
                    "--log-interval-eps", "4.0", "--mylabel", "somelabel", "--noise-ou",
                    "--noise-scale", "1.5", "--noise-scheduler", "linear_schedule",
                    "--range-output", "ranges.json", "--seed", "382500000", "--tbdir",
                    "tblog", "--total-timesteps", "500", "--version", "5.1")


@nox.session(python="3.8")
def looong(session):
    print(sys.version)
    common_install(session)
    with session.cd("source"):
        session.run("python", "-m",
                    "nb_exploration_metrics.colored_action_noise_deepdeterministic",
                    "--algorithm", "detsac", "--envname", "HopperPyBulletEnv-v0",
                    "--eval-interval-factor", "2.0", "--evaluate-ntimes", "100",
                    "--log-interval-eps", "4.0", "--mylabel", "somelabel", "--noise-ou",
                    "--noise-scale", "1.5", "--noise-scheduler", "linear_schedule",
                    "--range-output", "ranges.json", "--seed", "382500000", "--tbdir",
                    "tblog", "--version", "5.1")
