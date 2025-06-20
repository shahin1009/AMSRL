classdef RL_environment < rl.env.MATLABEnvironment
    %AMS_RL: Template for defining custom environment in MATLAB.    
    
    %% Properties (set properties' attributes accordingly)
    properties
        % Box count configuration - make this a primary property
        max_gen_boxes = 4; % Maximum number of boxes that can be in the system
        number_of_features=6;
        % Specify and initialize environment's necessary properties    
        % TODO: regulate the speeds
        v_treadmill = 0.6; % m/s

        d_AMS = 0.2; % m - size of the AMS
        n_i_AMS = 5; % numeber of AMS along y
        n_j_AMS = 5; % number of AMS along x
        n_actions_art = 2; % number of actions of each AMS

        upperlim_rot = pi*55/180; % upper limit rotation for AMS action
        upperlim_v = 2.5; % max velocity for AMS
        lowerlim_rot = - pi*55/180; % lower limit rotation for AMS action
        lowerlim_v = 0.4; % min velocity for AMS
        Big_TARG = 0.25; 
        Small_TARG = 0.6; 
        % Dynamic properties that depend on max_gen_boxes
        vy_boxes
        vx_boxes
        x_boxes_prec
        y_boxes_prec

        index_AMS

        LastAction % This will be resized based on AMS count

        i_box
        j_box

        n_boxes_tot_vect
        n_boxes_tot = 0
        cont_new_box = 0

        d_boxes
        x_boxes
        y_boxes
        x_boxes_vect
        y_boxes_vect
        cont = 0
        pack_exited
        exit_order
        index_exit = 0

        toll_contatto = 0.005

        dt = 0.01 % s - timestep for simulation

        time = 0 % s - simulation time
        order_reward = 0
        gap_penalty = 0
        collision_penalty = 0
        forward_reward = 0
        terminal_reward = 0
        step_sorting_reward = 0
        step_sorting_penalty = 0
        SizeThreshold = 0.225
        State
        SaveGIF = false
        GIFFile = 'sim.gif'
        GIFFrameCount = 1
        drawarrow = 1
    end

    

    properties (Hidden)
        % Flags for visualization
        VisualizeAnimation = true
        VisualizeActions = false
        VisualizeStates = false        
    end
   
    
    properties(Access = protected)
        % Initialize internal flag to indicate episode termination
        IsDone = false        
    end

    properties (Transient, Access = private)
        Visualizer = []
    end

    %% Necessary Methods
    methods              
        % Contructor method creates an instance of the environment
        % Change class name and constructor name accordingly
        function this = RL_environment()
            % Initialize Observation settings
           
            number_of_features=6;
            % ObservationInfo = ???; % states: to be defined
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            ObservationInfo=rlNumericSpec([number_of_features*10 1]);
            ObservationInfo.Name = 'System States';
            ObservationInfo.Description = 'Box positions, velocities, and AMS states';
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Initialize Action settings
            n_actions = 5*5*2;
            ActionInfo = rlNumericSpec([n_actions 1]);
            ActionInfo.Name = 'AMS Action';
            ActionInfo.Description = 'r1, v1, r2, v2, ...';
            ActionInfo.LowerLimit = zeros(n_actions,1);
            ActionInfo.UpperLimit = zeros(n_actions,1);
            this = this@rl.env.MATLABEnvironment(ObservationInfo,ActionInfo);
            this.initializeDependentProperties();
            for ii=1:2:n_actions
                
                ActionInfo.UpperLimit(ii) = this.upperlim_rot;
                ActionInfo.UpperLimit(ii+1) = this.upperlim_v;
                ActionInfo.LowerLimit(ii) = this.lowerlim_rot;
                ActionInfo.LowerLimit(ii+1) = this.lowerlim_v;

            end

            % The following line implements built-in functions of RL env

        end

        function initializeDependentProperties(this)
            % Initialize box state arrays based on max_gen_boxes
            this.vy_boxes = zeros(this.max_gen_boxes, 1);
            this.vx_boxes = zeros(this.max_gen_boxes, 1);
            this.x_boxes_prec = zeros(this.max_gen_boxes, 1);
            this.y_boxes_prec = zeros(this.max_gen_boxes, 1);

            this.index_AMS = zeros(this.max_gen_boxes, 1);

            this.i_box = zeros(this.max_gen_boxes, 1);
            this.j_box = zeros(this.max_gen_boxes, 1);

            this.n_boxes_tot_vect = zeros(1, this.max_gen_boxes);

            this.d_boxes = zeros(this.max_gen_boxes, 1);
            this.x_boxes = zeros(this.max_gen_boxes, 1);
            this.y_boxes = zeros(this.max_gen_boxes, 1);
            this.x_boxes_vect = zeros(this.max_gen_boxes, 10000);
            this.y_boxes_vect = zeros(this.max_gen_boxes, 10000);

            this.pack_exited = zeros(this.max_gen_boxes, 1);
            this.exit_order = zeros(this.max_gen_boxes, 1);

            this.State = zeros(this.number_of_features*10, 1);
        end

        % Apply system dynamics and simulates the environment with the
        % given action for one step.
        function [Observation,Reward,IsDone,Info] = step(this,Action)
            Info = [];

            this.time = this.time + this.dt;

            this.cont = this.cont + 1;

            l_AMS_matrix = this.d_AMS*this.n_j_AMS; % m

            ActLimUp = zeros(50,1);
            ActLimLow = zeros(50,1);

            for ii=1:2:50
                ActLimUp(ii) = this.upperlim_rot;
                ActLimUp(ii+1) = this.upperlim_v;
                ActLimLow(ii) = this.lowerlim_rot;
                ActLimLow(ii+1) = this.lowerlim_v;
            end

            % Actions are normalized [0-1]
            % De-normalizing the actions
            AMS_actions = ActLimLow + (1 + Action) .* (ActLimUp - ActLimLow)./2;
            for ii=1:50
                AMS_actions(ii) = max(ActLimLow(ii),min(ActLimUp(ii),AMS_actions(ii)));
            end

            this.LastAction = AMS_actions;

            v_AMS = zeros(5,5);
            rotation_AMS = zeros(5,5);

            for ii = 1:this.n_i_AMS
                for jj = 1:this.n_j_AMS
                    % rotational actions
                    rotation_AMS(ii,jj) = AMS_actions(jj*2-1+(ii-1)*10);
                    % velocity actions
                    v_AMS(ii,jj) = AMS_actions(jj*2+(ii-1)*10);
                end
            end

            % new boxes generation each 0.75 s. 2 boxes are generated.

            if mod(round(this.time,2),0.75) == 0 && this.n_boxes_tot<this.max_gen_boxes && this.time>0.25
                
                this.cont_new_box = this.cont_new_box + 1;
                %TODO: one box
                gen_n_boxes = 2;
                if gen_n_boxes>2
                    gen_n_boxes=2;
                end

                while this.n_boxes_tot+gen_n_boxes>this.max_gen_boxes
                    gen_n_boxes = gen_n_boxes-1;
                end

                this.n_boxes_tot = this.n_boxes_tot+gen_n_boxes;
                this.n_boxes_tot_vect(this.cont_new_box) = gen_n_boxes;

                for ii=1:gen_n_boxes

                    if gen_n_boxes>1
                        if ii == 1
                            this.x_boxes(this.n_boxes_tot-1) = this.d_boxes(this.n_boxes_tot-1)*0.5 + (this.d_AMS*2.5-this.d_boxes(this.n_boxes_tot-1)*0.5)*rand(1);
                            this.y_boxes(this.n_boxes_tot-1) = 0.001 + rand(1)*0.01;
                            this.x_boxes_prec(this.n_boxes_tot-1) = this.x_boxes(this.n_boxes_tot-1);
                            this.y_boxes_prec(this.n_boxes_tot-1) = this.y_boxes(this.n_boxes_tot-1);
                        else
                            this.x_boxes(this.n_boxes_tot) = this.d_AMS*2.5+this.d_boxes(this.n_boxes_tot)*0.5 + (l_AMS_matrix-(this.d_AMS*2.5+this.d_boxes(this.n_boxes_tot)*0.5))*rand(1);
                            this.y_boxes(this.n_boxes_tot) = 0.001 + rand(1)*0.05;
                            if abs(this.x_boxes(this.n_boxes_tot)-this.x_boxes(this.n_boxes_tot-1)) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                                this.x_boxes(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot-1) + this.d_boxes(this.n_boxes_tot)/2 + this.d_boxes(this.n_boxes_tot-1)/2 + 0.1;
                            end
                            if this.x_boxes(this.n_boxes_tot)-this.d_boxes(this.n_boxes_tot)/2<0
                                this.x_boxes(this.n_boxes_tot) = this.d_boxes(this.n_boxes_tot)/2;
                            elseif this.x_boxes(this.n_boxes_tot)+this.d_boxes(this.n_boxes_tot)/2>l_AMS_matrix
                                this.x_boxes(this.n_boxes_tot) = l_AMS_matrix-this.d_boxes(this.n_boxes_tot)/2;
                            end
                            if abs(this.x_boxes(this.n_boxes_tot)-this.x_boxes(this.n_boxes_tot-1)) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                                this.x_boxes(this.n_boxes_tot-1) = this.x_boxes(this.n_boxes_tot) - (this.d_boxes(this.n_boxes_tot)/2 + this.d_boxes(this.n_boxes_tot-1)/2 + 0.1);
                            end
                            this.x_boxes_prec(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot);
                            this.y_boxes_prec(this.n_boxes_tot) = this.y_boxes(this.n_boxes_tot);
                        end
                    else
                        this.x_boxes(this.n_boxes_tot) = this.d_boxes(this.n_boxes_tot)*0.55 + (l_AMS_matrix-this.d_boxes(this.n_boxes_tot)*0.55)*rand(1);
                        this.y_boxes(this.n_boxes_tot) = 0.001 + rand(1)*0.01;
                        this.x_boxes_prec(this.n_boxes_tot) = this.x_boxes(this.n_boxes_tot);
                        this.y_boxes_prec(this.n_boxes_tot) = this.y_boxes(this.n_boxes_tot);
                        this.x_boxes_vect(this.n_boxes_tot,this.cont-1) = this.x_boxes(this.n_boxes_tot);
                        this.y_boxes_vect(this.n_boxes_tot,this.cont-1) = this.y_boxes(this.n_boxes_tot);
                    end

                end

            end

            for ii=this.n_boxes_tot+1:this.max_gen_boxes
                this.x_boxes(ii) = -1;
                this.y_boxes(ii) = -1;
            end

            act_AMS = zeros(25,1);
            this.index_AMS = [];

            % checking collisions

            for ii=1:this.n_boxes_tot

                if this.x_boxes(ii)-this.d_boxes(ii)/2<0
                    this.x_boxes(ii) = this.d_boxes(ii)/2;
                elseif this.x_boxes(ii)+this.d_boxes(ii)/2>l_AMS_matrix
                    this.x_boxes(ii) = l_AMS_matrix-this.d_boxes(ii)/2;
                end

                % identifying on which AMS the boxes are (geometrical baricenter)

                if this.y_boxes(ii)<=this.d_AMS
                    this.i_box(ii) = 1;
                elseif this.y_boxes(ii)<=this.d_AMS*2
                    this.i_box(ii) = 2;
                elseif this.y_boxes(ii)<=this.d_AMS*3
                    this.i_box(ii) = 3;
                elseif this.y_boxes(ii)<=this.d_AMS*4
                    this.i_box(ii) = 4;
                elseif this.y_boxes(ii)<=this.d_AMS*5
                    this.i_box(ii) = 5;
                else
                    this.i_box(ii) = 6;
                end

                if this.i_box(ii) == 6
                    this.j_box(ii) = 6;
                elseif this.x_boxes(ii)<=this.d_AMS
                    this.j_box(ii) = 1;
                elseif this.x_boxes(ii)<=this.d_AMS*2
                    this.j_box(ii) = 2;
                elseif this.x_boxes(ii)<=this.d_AMS*3
                    this.j_box(ii) = 3;
                elseif this.x_boxes(ii)<=this.d_AMS*4
                    this.j_box(ii) = 4;
                else
                    this.j_box(ii) = 5;
                end

                % packages kinematics

                if this.i_box(ii) < 6
                    this.x_boxes(ii) = this.x_boxes(ii) + v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*sin(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                    this.y_boxes(ii) = this.y_boxes(ii) + v_AMS(this.i_box(ii),this.j_box(ii))*this.dt*cos(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                    this.vy_boxes(ii) = v_AMS(this.i_box(ii),this.j_box(ii))*cos(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                    this.vx_boxes(ii) = v_AMS(this.i_box(ii),this.j_box(ii))*sin(rotation_AMS(this.i_box(ii),this.j_box(ii)));
                
                else
                    this.x_boxes(ii) = this.x_boxes(ii);
                    this.y_boxes(ii) = this.y_boxes(ii) + this.v_treadmill*this.dt;
                    this.vy_boxes(ii) = this.v_treadmill;
                    this.vx_boxes(ii) = 0;
                end

                % collisions

                if ii>1 && this.cont>1
                    for jj=ii:-1:2
                        if abs(this.x_boxes(ii)-this.x_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 + 0.001 ...
                                && abs(this.x_boxes_vect(ii,this.cont-1)-this.x_boxes_vect(jj-1,this.cont-1)) >= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.y_boxes(ii)-this.y_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.y_boxes_vect(ii,this.cont-1)-this.y_boxes_vect(jj-1,this.cont-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2
                            if (this.x_boxes(ii)-this.d_boxes(ii)/2 <= this.x_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.x_boxes(ii)-this.d_boxes(ii)/2>this.x_boxes(jj-1)-this.d_boxes(jj-1)/2)

                                penetrazione_x = (this.x_boxes(jj-1)+this.d_boxes(jj-1)/2) - (this.x_boxes(ii)-this.d_boxes(ii)/2);
                                this.x_boxes(ii) = this.x_boxes(ii) + penetrazione_x/2 + this.toll_contatto;
                                this.x_boxes(jj-1) = this.x_boxes(jj-1) - penetrazione_x/2 - this.toll_contatto;

                            elseif (this.x_boxes(ii)+this.d_boxes(ii)/2 >= this.x_boxes(jj-1)-this.d_boxes(jj-1)/2 && this.x_boxes(ii)+this.d_boxes(ii)/2<this.x_boxes(jj-1)+this.d_boxes(jj-1)/2)

                                penetrazione_x = (this.x_boxes(ii)+this.d_boxes(ii)/2) - (this.x_boxes(jj-1)-this.d_boxes(jj-1)/2);
                                this.x_boxes(ii) = this.x_boxes(ii) - penetrazione_x/2 - this.toll_contatto;
                                this.x_boxes(jj-1) = this.x_boxes(jj-1) + penetrazione_x/2 + this.toll_contatto;

                            end
                        elseif abs(this.y_boxes(ii)-this.y_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 + 0.001 ...
                                && abs(this.y_boxes_vect(ii,this.cont-1)-this.y_boxes_vect(jj-1,this.cont-1)) >= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                                && abs(this.x_boxes(ii)-this.x_boxes(jj-1)) <= this.d_boxes(ii)/2+this.d_boxes(jj-1)/2 ...
                            if (this.y_boxes(ii)-this.d_boxes(ii)/2 <= this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.y_boxes(ii)-this.d_boxes(ii)/2>this.y_boxes(jj-1)-this.d_boxes(jj-1)/2) % || (this.y_boxes(jj-1)-this.d_boxes(jj-1)/2 <= this.y_boxes(ii)+this.d_boxes(ii)/2 && this.y_boxes(jj-1)-this.d_boxes(jj-1)/2>this.y_boxes(ii)-this.d_boxes(ii)/2)
                                penetrazione_y = (this.y_boxes(jj-1)+this.d_boxes(jj-1)/2)-(this.y_boxes(ii)-this.d_boxes(ii)/2);
                                this.y_boxes(ii) = this.y_boxes(ii) + penetrazione_y/2 + this.toll_contatto;
                                this.y_boxes(jj-1) = this.y_boxes(jj-1) - penetrazione_y/2 - this.toll_contatto;
                            elseif (this.y_boxes(ii)+this.d_boxes(ii)/2 < this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 && this.y_boxes(ii)+this.d_boxes(ii)/2 >= this.y_boxes(jj-1)-this.d_boxes(jj-1)/2) % || (this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 < this.y_boxes(ii)+this.d_boxes(ii)/2 && this.y_boxes(jj-1)+this.d_boxes(jj-1)/2 >= this.y_boxes(ii)-this.d_boxes(ii)/2)
                                penetrazione_y = (this.y_boxes(ii)+this.d_boxes(ii)/2) - (this.y_boxes(jj-1)-this.d_boxes(jj-1)/2);
                                this.y_boxes(ii) = this.y_boxes(ii) - penetrazione_y/2  - this.toll_contatto;
                                this.y_boxes(jj-1) = this.y_boxes(jj-1) + penetrazione_y/2 + this.toll_contatto;
                            end
                        end
                    end
                end
                
                if this.i_box(ii)<6 && this.j_box(ii)<6
                    act_AMS(this.j_box(ii)+(this.i_box(ii)-1)*5) = 1;
                    this.index_AMS(ii) = this.j_box(ii)+(this.i_box(ii)-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end
            end


            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Observation: observations for the RL to be defined
            
            % Create observation vector
            Observation = zeros(this.number_of_features*10, 1);
            
            % First set: box positions and properties
            for ii = 1:this.n_boxes_tot
                % Box x-positions (indices 1-10)
                if ~this.pack_exited(ii)
                    Observation(ii) = this.x_boxes(ii);

                    % Box y-positions (indices 11-20)
                    Observation(10+ii) = this.y_boxes(ii);

                    Observation(20+ii) = this.d_boxes(ii);


                    if this.d_boxes(ii) < this.SizeThreshold
                        target_x = this.Small_TARG;
                    else
                        target_x = this.Big_TARG;
                    end
                    Observation(50+ii) = (target_x - this.x_boxes(ii));
                    Observation(30+ii)=this.vy_boxes(ii);
                    Observation(40+ii)=this.vx_boxes(ii);
                    
                else
                    Observation(ii) = 0;

                   
                    Observation(10+ii) = 0;

                  
                    Observation(20+ii) = 0;
                    Observation(30+ii)=0;
                    Observation(40+ii)=0;

                    Observation(50+ii) = 0;
                 
                
                end

            end
            
            

            
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            for ii=1:this.max_gen_boxes
                if ii<=this.n_boxes_tot
                    this.x_boxes_vect(ii,this.cont) = this.x_boxes(ii);
                    this.y_boxes_vect(ii,this.cont) = this.y_boxes(ii);
                    this.x_boxes_prec(ii) = this.x_boxes(ii);
                    this.y_boxes_prec(ii) = this.y_boxes(ii);
                else
                    this.x_boxes_vect(ii,this.cont) = -1;
                    this.y_boxes_vect(ii,this.cont) = -1;
                end
            end

            % Update system states
            this.State = Observation;

            cont_exit = 0;

            % Check terminal condition
            for ii=1:this.n_boxes_tot
                if this.y_boxes(ii)>this.d_AMS*5
                    cont_exit = cont_exit + 1;
                    if this.pack_exited(ii) == 0
                        this.pack_exited(ii) = 1;
                        this.index_exit = this.index_exit + 1;
                        this.exit_order(this.index_exit) = ii;
                    end
                end
            end

            if cont_exit==this.max_gen_boxes
                clc;
                IsDone = true;
                fprintf('order_reward:           %.2f\n', this.order_reward);
                fprintf('gap_penalty:            %.2f\n', this.gap_penalty);
                fprintf('collision_penalty:      %.2f\n', this.collision_penalty);
                fprintf('forward_reward:         %.2f\n', this.forward_reward);
                fprintf('terminal_reward:        %.2f\n', this.terminal_reward);
                fprintf('step_sorting_reward:    %.2f\n', this.step_sorting_reward);
                fprintf('step_sorting_penalty:   %.2f\n', this.step_sorting_penalty);
                fprintf('===============end of episode============= \n')
                disp('sizes')
                disp(this.d_boxes(1:this.max_gen_boxes).')
                disp('exit_x')
                disp(this.x_boxes_prec.')

                SIZE_THRESHOLD = this.SizeThreshold;       % Example size threshold
                Big_TARGET = this.Big_TARG;            % Example target for big boxes
                Small_TARGET = this.Small_TARG;           % Example target for small boxes
                DIST_THRESHOLD = 0.08;       % Distance threshold

                % Extract values
                diameters = this.d_boxes(1:this.max_gen_boxes).';
                binary_vector = zeros(size(diameters));

                for i = 1:length(diameters)
                    if diameters(i) > SIZE_THRESHOLD
                        target_x = Big_TARGET;
                    else
                        target_x = Small_TARGET;
                    end

                    % Check if distance to target is below the threshold
                    if abs(this.x_boxes_prec(i) - target_x) < DIST_THRESHOLD
                        binary_vector(i) = 1;
                    else
                        binary_vector(i) = 0;
                    end
                end

                disp(binary_vector)
                fprintf('===============end of episode============= \n')

            else
                IsDone = false;
            end

            this.IsDone = IsDone;

            
            % Get reward
            Reward = getReward(this,AMS_actions);
            
           
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);
        end
        
        % Reset environment to initial state and output initial observation
        % for each episod
        function InitialObservation = reset(this)
            this.order_reward=0;
            this.gap_penalty =0;
            this.collision_penalty =0;
            this.forward_reward=0;
            this.terminal_reward=0;
            this.step_sorting_reward=0;
            this.step_sorting_penalty=0;

            this.index_exit = 0;
            this.pack_exited = zeros(this.max_gen_boxes,1); % is package exited the AMS or not?
            this.exit_order = zeros(this.max_gen_boxes,1); % packages ordered by exit

            this.cont_new_box = 1;
            %TODO: one box
            this.n_boxes_tot_vect =2;
            this.n_boxes_tot = this.n_boxes_tot_vect;
            this.x_boxes_vect = zeros(this.max_gen_boxes,10000);
            this.y_boxes_vect = zeros(this.max_gen_boxes,10000);

            this.time = 0;

            this.vy_boxes = zeros(this.max_gen_boxes,1);
            this.vx_boxes = zeros(this.max_gen_boxes,1);

            this.cont = 0;

            max_d_box = 2.*this.d_AMS; % m
            min_d_box = 0.25*this.d_AMS; % m

            l_AMS_matrix = this.d_AMS*this.n_j_AMS; % m

            % generating initial boxes (2)
            for ii=1:this.max_gen_boxes
                this.d_boxes(ii) = min_d_box + (max_d_box-min_d_box)*rand(1);
                % if mod(ii, 2) == 1
                %   this.d_boxes(ii) = 0.225 + (max_d_box-0.225)*rand(1);
                % 
                % % else
                % %     this.d_boxes(ii) = min_d_box + (0.225-min_d_box)*rand(1);
                % end
            end
            %TODO: one box
            for ii=1:2
                if ii == 1
                    x1 = this.d_boxes(1)*0.5 + (this.d_AMS*2.5-this.d_boxes(1)*0.5)*rand(1);
                else
                    x2 = this.d_AMS*2.5+this.d_boxes(2)*0.5 + (l_AMS_matrix-(this.d_AMS*2.5+this.d_boxes(2)*0.5))*rand(1);
                    if abs(x2-x1) <= this.d_boxes(this.n_boxes_tot)/2+this.d_boxes(this.n_boxes_tot-1)/2
                        x2 = x1 + this.d_boxes(2)/2 + this.d_boxes(1)/2 + 0.1;
                    end
                    if x2-this.d_boxes(2)/2<0
                        x2 = this.d_boxes(2)/2;
                    elseif x2+this.d_boxes(2)/2>l_AMS_matrix
                        x2 = l_AMS_matrix-this.d_boxes(2)/2;
                    end
                    if abs(x2-x1) <= this.d_boxes(2)/2+this.d_boxes(1)/2
                        x1 = x2 - (this.d_boxes(2)/2 + this.d_boxes(1)/2 + 0.1);
                    end
                end
                y1 = 0.001 + rand(1)*0.01;
                y2 = 0.001 + rand(1)*0.05;
            end
            
           

            this.x_boxes(1) = x1;
            this.y_boxes(1) = y1;
            this.x_boxes(2) = x2;
            this.y_boxes(2) = y2;
           

            this.x_boxes_prec(1) = x1;
            this.y_boxes_prec(1) = y1;
            this.x_boxes_prec(2) = x2;
            this.y_boxes_prec(2) = y2;

            for ii=this.n_boxes_tot+1:this.max_gen_boxes
                this.x_boxes(ii) = 0;
                this.y_boxes(ii) = 0;
            end

            act_AMS = zeros(25,1);
            this.index_AMS = zeros(2,1);

            % identifying on which AMS the boxes are (geometrical
            % baricenter)

            for ii=1:this.n_boxes_tot

                if this.y_boxes(ii)<=this.d_AMS
                    this.i_box(ii) = 1;
                elseif this.y_boxes(ii)<=this.d_AMS*2
                    this.i_box(ii) = 2;
                elseif this.y_boxes(ii)<=this.d_AMS*3
                    this.i_box(ii) = 3;
                elseif this.y_boxes(ii)<=this.d_AMS*4
                    this.i_box(ii) = 4;
                elseif this.y_boxes(ii)<=this.d_AMS*5
                    this.i_box(ii) = 5;
                else
                    this.i_box(ii) = 6;
                end

                if this.i_box(ii) == 6
                    this.j_box(ii) = 6;
                elseif this.x_boxes(ii)<=this.d_AMS
                    this.j_box(ii) = 1;
                elseif this.x_boxes(ii)<=this.d_AMS*2
                    this.j_box(ii) = 2;
                elseif this.x_boxes(ii)<=this.d_AMS*3
                    this.j_box(ii) = 3;
                elseif this.x_boxes(ii)<=this.d_AMS*4
                    this.j_box(ii) = 4;
                else
                    this.j_box(ii) = 5;
                end

                if this.i_box(ii)<6 && this.j_box(ii)<6
                    act_AMS(this.j_box(ii)+(this.i_box(ii)-1)*5) = 1;
                    this.index_AMS(ii) = this.j_box(ii)+(this.i_box(ii)-1)*5;
                else
                    this.index_AMS(ii) = 0;
                end

            end

            % InitialObservation: initial observation for the RL to be
            % defined
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Create initial observation vector
            InitialObservation = zeros(this.number_of_features*10, 1);
            
            % First set: box positions and properties
            for ii = 1:this.n_boxes_tot
               
                    % Box x-positions (indices 1-10)
                    InitialObservation(ii) = this.x_boxes(ii);

                    % Box y-positions (indices 11-20)
                    InitialObservation(10+ii) = this.y_boxes(ii);

                    % Box dimensions (indices 21-30)
                    InitialObservation(20+ii) = this.d_boxes(ii);


                    if this.d_boxes(ii) < this.SizeThreshold % Large box
                        target_x = this.Small_TARG;
                    else
                        target_x = this.Big_TARG;
                    end
                    InitialObservation(50+ii) = (target_x - this.x_boxes(ii));
                    InitialObservation(30+ii) = this.vy_boxes(ii);
                    InitialObservation(40+ii) = this.vx_boxes(ii);
               
            end
            
            % Box velocities (indices 31-40)
            for ii = 1:length(this.vy_boxes)
                if ii <= this.n_boxes_tot
                    InitialObservation(30+ii) = this.vy_boxes(ii);
                    InitialObservation(40+ii) = this.vx_boxes(ii);
                end
            end
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            this.State = InitialObservation;
            
            % (optional) use notifyEnvUpdated to signal that the 
            % environment has been updated (e.g. to update visualization)
            notifyEnvUpdated(this);

        end
    end
    %% Optional Methods (set methods' attributes accordingly)
    methods  

        function varargout = plot(this)
            if isempty(this.Visualizer) || ~isvalid(this.Visualizer)
                this.Visualizer = AMSVisualizer(this);
            else
                bringToFront(this.Visualizer);
            end
            if nargout
                varargout{1} = this.Visualizer;
            end
            % Reset Visualizations
            this.VisualizeAnimation = true;
            this.VisualizeActions = false;
            this.VisualizeStates = false;
        end

       
        function Reward = getReward(this, Action)
            
            
            SIZE_THRESHOLD = this.SizeThreshold;
            


            l_AMS_matrix = this.d_AMS * this.n_j_AMS;

            Big_TARGET = this.Big_TARG; 
            Small_TARGET = this.Small_TARG; 
      
            EXIT_Y_THRESHOLD = 0.9 * (this.d_AMS * this.n_i_AMS);  
            ALIGNMENT_THRESHOLD = 0.08; 

            active_boxes = find(~this.pack_exited(1:this.n_boxes_tot));

            
            total_reward = 0;
            total_forward = 0;
            total_alignment = 0;
            total_exit_bonus = 0;
         
            alignment_reward=0;
            vx_reward = 0;
            for i = 1:numel(active_boxes)
                % Get box properties
                box_idx = active_boxes(i);
                box_size = this.d_boxes(box_idx);
                x_pos = this.x_boxes(box_idx);
                y_pos = this.y_boxes(box_idx);
                if box_size > SIZE_THRESHOLD
                    target_x = Big_TARGET; 
               
                    
                else
                    target_x = Small_TARGET;
                 
                end
                ALIGNMENT_WEIGHT = 3;
                EXIT_BONUS_WEIGHT = 16.0;
                if ~this.pack_exited(box_idx)
                    dist=target_x - x_pos;
                    % ----- Component 1: Lane Alignment Reward -----
                    distance = abs(dist);

                    diff = distance-ALIGNMENT_THRESHOLD;

                    
                    if diff>0
                        diff=diff*10;

                    end
                    alignment_reward = ALIGNMENT_WEIGHT * (- diff);
                    total_alignment = total_alignment + y_pos*alignment_reward;
                  


                    % ----- Component 3: Exit Bonus -----

                    if y_pos>EXIT_Y_THRESHOLD
                        if distance < ALIGNMENT_THRESHOLD
                            alignment_quality = 1 - (distance / ALIGNMENT_THRESHOLD);
                            total_exit_bonus = total_exit_bonus + EXIT_BONUS_WEIGHT * alignment_quality;
                        else
                            alignment_quality = - (distance / ALIGNMENT_THRESHOLD);
                            total_exit_bonus = total_exit_bonus + EXIT_BONUS_WEIGHT * alignment_quality/10;

                        end



                    end

                    if y_pos>0.5
                        vx = this.vx_boxes(box_idx);
                        
                        if distance>0.3
                            vx_reward=vx_reward+sign(dist)*vx*5;
                        end
                        
                    end

                end
                
            end

            gap_penalty_total=0;
            vy_reward = 0;
            GAP_WEIGHT=6;
            Required_distance=0.2;
            if numel(active_boxes)>1
                for i = 1:numel(active_boxes)-1
                    for j = i+1:numel(active_boxes)

                        box1 = active_boxes(i);
                        box2 = active_boxes(j);

                        if this.y_boxes(box2) > 0.25
                            gap = abs(this.y_boxes(box1) - this.y_boxes(box2));
                            penalty = GAP_WEIGHT / (1 + exp(25*(gap - Required_distance)));
                            gap_penalty_total = gap_penalty_total + penalty;
                        end

                    end
                end
                avg_vy = mean(this.vy_boxes);
                vy_reward = vy_reward+0*avg_vy;
            end
            Final_reward=0;
           
            if all(this.pack_exited(1:this.max_gen_boxes))

                for i = 1:length(this.x_boxes_prec(1:this.max_gen_boxes))

                    if this.d_boxes(i) > SIZE_THRESHOLD && abs(this.x_boxes_prec(i) - this.Big_TARG)<ALIGNMENT_THRESHOLD
                      Final_reward=Final_reward+75;
                      
                    elseif this.d_boxes(i) < SIZE_THRESHOLD && abs(this.x_boxes_prec(i) - this.Small_TARG)<ALIGNMENT_THRESHOLD
                      Final_reward=Final_reward+75;
                     
                    end
                end
            
            end
            

            % Update reward tracking properties
            this.forward_reward = this.forward_reward+Final_reward;
            this.gap_penalty = this.gap_penalty -gap_penalty_total;
            this.step_sorting_reward = this.step_sorting_reward + total_alignment;
            this.step_sorting_penalty=this.step_sorting_penalty+vx_reward;
            this.terminal_reward = this.terminal_reward + total_exit_bonus;
            % You may want to add a new property to track direction rewards

            Reward = vx_reward-gap_penalty_total+total_exit_bonus+total_alignment+Final_reward;
           
        end

        function set.State(this,state)
            validateattributes(state,{'numeric'},{'finite','real','vector','numel',this.number_of_features*10},'','State');
            this.State = double(state(:));
            notifyEnvUpdated(this);
        end

    end

    methods (Access = protected)
        % (optional) update visualization everytime the environment is updated 
        % (notifyEnvUpdated is called)
        function envUpdatedCallback(this)
        end
    end
end