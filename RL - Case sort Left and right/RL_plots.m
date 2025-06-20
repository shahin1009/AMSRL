function RL_plots(ax, experiences, env, name)
% plotArtichokes plots the time history of states of the
% artichokes system.
%
% ax is a MATLAB axes object.
% experiences is the output of the sim command.
% env is the environment object.
% name is the name of the observation.

% Copyright 2023 The MathWorks Inc.

arguments
    ax (1,1) matlab.graphics.axis.Axes
    experiences struct
    env (1,1) artichokes_RL_mod
    name string {mustBeMember(name,["x1", "y1", "x2", "y2"])}
end

observations = [experiences.Observation];
ts = [observations.PackagesStates];
obsInfo = getObservationInfo(env);
numObs = obsInfo.Dimension(1);

hold(ax, "on");

for ct = 1:numel(ts)

    obs = reshape(ts(ct).Data, numObs, ts(ct).Length)';
    
    switch lower(name)
        case "x1"
            ts(ct)
            plot(ax, ts(ct).Time(1:env.cont), env.x_pacchi_vect(1:env.cont,1));
            xlabel(ax,"Time (s)");
            ylabel(ax,"x1");
            title(ax,"x1 (m)");

        case "y1"
            plot(ax, ts(ct).Time(1:env.cont), env.y_pacchi_vect(1:env.cont,1));
            xlabel(ax,"Time (s)");
            ylabel(ax,"y1");
            title(ax,"y1 (m)");

        case "x2"
            plot(ax, ts(ct).Time(1:env.cont), env.x_pacchi_vect(1:env.cont,2));
            xlabel(ax,"Time (s)");
            ylabel(ax,"x2");
            title(ax,"x2 (m)");

        case "y2"
            plot(ax, ts(ct).Time(1:env.cont), env.y_pacchi_vect(1:env.cont,2));
            xlabel(ax,"Time (s)");
            ylabel(ax,"y2");
            title(ax,"y2 (m)");
    end
end
end