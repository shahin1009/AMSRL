
clear all
close all
clc

%%

% load the RL environment

env=RL_environment();

UseParallel = true;
UseGpu = true;
istrained = false;
% env.visual = true;

if istrained == false

    actInfo = getActionInfo(env);
    obsInfo = getObservationInfo(env);

    numObs = prod(obsInfo.Dimension);
    criticLayerSizes = [32 32 16];
    actorLayerSizes = [32 32 16];

    % Critic

    criticNetwork = [
        featureInputLayer(numObs)
        fullyConnectedLayer(criticLayerSizes(1))
        reluLayer
        fullyConnectedLayer(criticLayerSizes(2))
        reluLayer
        fullyConnectedLayer(criticLayerSizes(3))
        reluLayer
        fullyConnectedLayer(1)
        ];

    criticNetwork = dlnetwork(criticNetwork);
    summary(criticNetwork)

    critic = rlValueFunction(criticNetwork,obsInfo);
    
    % Actor

    inPath = [
        featureInputLayer(numObs,Name="netOin")
        fullyConnectedLayer(actorLayerSizes(1))
        reluLayer
        fullyConnectedLayer(actorLayerSizes(2))
        reluLayer(Name="relulast")
        ];

    meanPath = [
        fullyConnectedLayer(actorLayerSizes(3),Name="MeanLyr")
        reluLayer
        fullyConnectedLayer(prod(actInfo.Dimension),Name="meanOutLyr")
        tanhLayer(Name="thmeanOutLyr");
        ];

    sdevPath = [
        fullyConnectedLayer(actorLayerSizes(3),Name="StdLyr")
        reluLayer
        fullyConnectedLayer(prod(actInfo.Dimension))
        reluLayer
        softplusLayer(Name="stdOutLyr")
        ];

    % Add layers to network object
    net = layerGraph(inPath);
    net = addLayers(net,meanPath);
    net = addLayers(net,sdevPath);

    % Connect layers
    net = connectLayers(net,"relulast","MeanLyr/in");
    net = connectLayers(net,"relulast","StdLyr/in");

    net = dlnetwork(net);
    summary(net)

    actor = rlContinuousGaussianActor(net, obsInfo, actInfo, ...
        ActionMeanOutputNames="thmeanOutLyr",...
        ActionStandardDeviationOutputNames="stdOutLyr",...
        ObservationInputNames="netOin");
    
    % Train

    actorOpts = rlOptimizerOptions(LearnRate=1e-3);
    criticOpts = rlOptimizerOptions(LearnRate=1e-3);
    
    if UseGpu
        critic.UseDevice = "gpu";
        actor.UseDevice = "gpu";
    else
        critic.UseDevice = "cpu";
        actor.UseDevice = "cpu";
    end

    agentOpts = rlPPOAgentOptions(...
        ExperienceHorizon=512,...
        ClipFactor=0.1,...
        EntropyLossWeight=0.001,...
        ActorOptimizerOptions=actorOpts,...
        CriticOptimizerOptions=criticOpts,...
        NumEpoch=3,...
        AdvantageEstimateMethod="gae",...
        GAEFactor=0.95,...
        SampleTime=0.01,...
        DiscountFactor=0.99, ...
        MiniBatchSize=512);

    agent = rlPPOAgent(actor,critic,agentOpts);

    trainOpts = rlTrainingOptions(...
        MaxEpisodes=5000,...
        MaxStepsPerEpisode=1000,...
        Plots = "none",... %Plots="training-progress",...
        StopTrainingCriteria="AverageReward",...
        StopTrainingValue=40000,...
        ScoreAveragingWindowLength=100, ...
        UseParallel=UseParallel,Verbose=true);

    trainingStats = train(agent, env, trainOpts);

    save("agent_trained","agent");
    save("trainingStats","trainingStats");





else
    % load("agent_trained_gap_v2.mat")
    % load("trainingStats_gap_v2.mat")
    env = RL_environment();
    trainOpts = rlTrainingOptions(...
        MaxEpisodes=2000,...
        MaxStepsPerEpisode=1000,...
        Plots = "none",...%Plots="training-progress",...
        StopTrainingCriteria="AverageReward",...
        StopTrainingValue=40000,...
        ScoreAveragingWindowLength=100, ...
        UseParallel=UseParallel,Verbose=true);

    trainingStats = train(agent, env, trainOpts);

end

%% simulation of the sorting after training
env = RL_environment();
env.reset();

plot(env)

rng(10)
simOptions = rlSimulationOptions(MaxSteps=10000);
simOptions.NumSimulations = 10;
experience = sim(env, agent, simOptions);
