
clear all
close all
clc

%%

% load the RL environment

env=RL_environment();

actInfo = getActionInfo(env);
obsInfo = getObservationInfo(env);

numObs = prod(obsInfo.Dimension);
criticLayerSizes = [64 32 32];
actorLayerSizes = [64 32 32];

%% Critic

criticNetwork = [
    featureInputLayer(numObs)
    fullyConnectedLayer(criticLayerSizes(1), ...
        Weights=sqrt(2/numObs)*...
            (rand(criticLayerSizes(1),numObs)-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(1),1))
    reluLayer
    fullyConnectedLayer(criticLayerSizes(2), ...
        Weights=sqrt(2/criticLayerSizes(1))*...
            (rand(criticLayerSizes(2),criticLayerSizes(1))-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(2),1))
    reluLayer
    fullyConnectedLayer(criticLayerSizes(3), ...
        Weights=sqrt(2/criticLayerSizes(2))*...
            (rand(criticLayerSizes(3),criticLayerSizes(2))-0.5), ...
        Bias=1e-3*ones(criticLayerSizes(3),1))
    reluLayer
    fullyConnectedLayer(1, ...
        Weights=sqrt(2/criticLayerSizes(3))* ...
            (rand(1,criticLayerSizes(3))-0.5), ...
        Bias=1e-3)
    ];

criticNetwork = dlnetwork(criticNetwork);
summary(criticNetwork)

critic = rlValueFunction(criticNetwork,obsInfo);

%% Actor

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

%% Train

actorOpts = rlOptimizerOptions(LearnRate=1e-4);
criticOpts = rlOptimizerOptions(LearnRate=1e-4);

newActorOpts = rlOptimizerOptions(LearnRate=1e-6); 
newCriticOpts = rlOptimizerOptions(LearnRate=1e-6);

agentOpts = rlPPOAgentOptions(...
    ExperienceHorizon=500,...
    ClipFactor=0.01,...
    EntropyLossWeight=0.001,...
    ActorOptimizerOptions=actorOpts,...
    CriticOptimizerOptions=criticOpts,...
    NumEpoch=3,...
    AdvantageEstimateMethod="gae",...
    GAEFactor=0.95,...
    SampleTime=0.01,...
    DiscountFactor=0.995);

agentOpts.ActorOptimizerOptions = newActorOpts;
agentOpts.CriticOptimizerOptions = newCriticOpts;

agent = rlPPOAgent(actor,critic,agentOpts);
%%
trainOpts = rlTrainingOptions(...
    MaxEpisodes=5000,...
    MaxStepsPerEpisode=1000,...
    Plots="training-progress",...
    StopTrainingCriteria="AverageReward",...
    StopTrainingValue=40000,...
    ScoreAveragingWindowLength=100);

%%
load('agent_trained.mat','agent')
actor = getActor(agent);
critic = getCritic(agent);
env=RL_environment();
%%

trainingStats = train(agent, env, trainOpts);

%%
save("agent_trained","agent");
save("trainingStats","trainingStats");
% 
%% simulation after training

env.reset();

plot(env)
rng(20)
env.SaveGIF = true;
env.GIFFile = 'sim.gif';
env.GIFFrameCount = 1;
env.drawarrow = 0;
simOptions = rlSimulationOptions(MaxSteps=10000);
simOptions.NumSimulations = 5;
experience = sim(env, agent, simOptions);
%%
distance_to_target =[];
for i =1:25
    Data_=squeeze(experience(i).Observation.SystemStates.Data);
    for j=1:4
        [v,ind]=max(Data_(10+j,:));
        distance_to_target=[distance_to_target,Data_(50+j,ind)];

    end
end

x = 1:100;
y = abs(distance_to_target);
figure;

scatter(x, y, 40, y, 'filled'); hold on;
plot(1:100,abs(distance_to_target),Color='blue')
title('deviation from target')
grid on
xlabel('Box Id')
ylabel('distance to the target')
ylim([0,0.3])
yline(0.08,Color='green',LineWidth=2)