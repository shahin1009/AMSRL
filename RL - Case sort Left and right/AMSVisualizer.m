classdef AMSVisualizer < rl.env.viz.AbstractFigureVisualizer

    properties (Access = private)
        BoxHandles
        BoxTextHandles
        ArrowHandles  
    end
    

    methods
        function this = AMSVisualizer(env)
            this = this@rl.env.viz.AbstractFigureVisualizer(env);
        end
    end

    methods (Access = protected)
        function f = buildFigure(this)
            f = figure( ...
                Toolbar="none", ...
                Visible="on", ...
                HandleVisibility="off", ...
                NumberTitle="off", ...
                Name="AMS",...
                CloseRequestFcn=@(~,~)delete(this));
            if ~strcmp(f.WindowStyle, "docked")
                f.Position = [200 100 800 500];
            end
            f.MenuBar = "none";

            ha = gca(f);
            ha.XLimMode = "manual";
            ha.YLimMode = "manual";
            ha.ZLimMode = "manual";
            ha.DataAspectRatioMode = "manual";
            ha.PlotBoxAspectRatioMode = "manual";
            ha.XLim = [0 1];
            ha.YLim = [0 1];
            hold(ha,"on");
        end

        function updatePlot(this)

            env = this.Environment;

            action = env.LastAction;
            rot_AMS = zeros(5,5);
            vel_AMS = zeros(5,5);

            upperlim_v = env.upperlim_v;
            lowerlim_v = env.lowerlim_v;
            if action
                for ii = 1:5
                    for jj = 1:5
                        rot_AMS(ii,jj) = action(jj*2 - 1 + (ii-1)*10);
                        v_norm = action(jj*2 + (ii-1)*10)/(upperlim_v - lowerlim_v) - lowerlim_v;
                        vel_AMS(ii,jj) = 0.05 + v_norm * (0.15 - 0.05);
                    end
                end
            end
            f = this.Figure;
            ha = gca(f);
            d_AMS = env.d_AMS;
            numBoxes = env.max_gen_boxes;

            % Preallocate box data arrays
            d_boxes = zeros(1, numBoxes);
            x_boxes = zeros(1, numBoxes);
            y_boxes = zeros(1, numBoxes);

            for ii = 1:numBoxes
                d_boxes(ii) = env.d_boxes(ii);
                x_boxes(ii) = env.x_boxes(ii);
                y_boxes(ii) = env.y_boxes(ii);
            end

          
            % Draw grid lines (horizontal and vertical) if not already there
            existingLines = findobj(ha, 'Type', 'line');
            if isempty(existingLines)
                for i = 0:5
                    y = d_AMS * i;
                    line(ha, ha.XLim, [y y], 'LineWidth', 2, 'Color', 'k', 'Tag', sprintf('h%uplot', i+1));
                    x = d_AMS * i;
                    line(ha, [x x], ha.YLim, 'LineWidth', 2, 'Color', 'k', 'Tag', sprintf('v%uplot', i+1));
                end
                
                % line(ha, [0.0 0.5], [1 1], 'LineWidth', 3, 'Color', 'b', 'Tag', 'leftHline');   % Blue line
                % line(ha, [0.5 1], [1 1], 'LineWidth', 3, 'Color', 'g', 'Tag', 'rightHline');  % Green (or change as desired)
                % line(ha, [0.5 0.5], ha.YLim, 'LineWidth', 4, 'Color', 'r', 'Tag', 'x05line');
            end
            if env.drawarrow

                % Create or update arrows for actuation visualization
                if isempty(this.ArrowHandles) || length(this.ArrowHandles) ~= 25 || any(~isgraphics(this.ArrowHandles))
                    % Initialize arrow handles if not already done
                    if ishandle(this.ArrowHandles)
                        delete(this.ArrowHandles);
                    end
                    this.ArrowHandles = gobjects(5, 5);
                end

                % Define the grid spacing
                grid_size = d_AMS;

                % Update arrows to show rotation and velocity for each square in the grid
                for ii = 1:5
                    for jj = 1:5
                        % Calculate center of the current grid cell
                        centerX = (jj - 0.5) * grid_size;
                        centerY = (ii - 0.5) * grid_size;

                        % Get rotation and velocity for this cell
                        rotation = rot_AMS(ii, jj);  % Angle in radians
                        velocity = vel_AMS(ii, jj);  % Magnitude of velocity

                        % Calculate arrow endpoint using polar coordinates
                        arrowLength = velocity * grid_size*2 ;  % Scale velocity to a reasonable arrow length
                        dx = arrowLength * sin(rotation);
                        dy = arrowLength * cos(rotation);

                        % Delete existing arrow if it exists
                        if isgraphics(this.ArrowHandles(ii, jj))
                            delete(this.ArrowHandles(ii, jj));
                        end

                        % Create arrow to show direction and magnitude
                        if action  % Only draw arrows if there's an action
                            this.ArrowHandles(ii, jj) = quiver(ha, centerX, centerY, dx, dy, 0, ...
                                'LineWidth', 2, ...
                                'MaxHeadSize', 1, ...
                                'Color', 'r', ...
                                'Tag', sprintf('arrow_%d_%d', ii, jj));
                        end
                    end
                end
            end
            % Create rectangles if not already initialized
            if isempty(this.BoxHandles) || any(~isgraphics(this.BoxHandles))
                this.BoxHandles = gobjects(1, numBoxes);
                colors = lines(numBoxes); % Distinct colors
                for i = 1:numBoxes
                    this.BoxHandles(i) = rectangle(ha, ...
                        'Position', [x_boxes(i), y_boxes(i), d_boxes(i)/2, d_boxes(i)/2], ...
                        'FaceColor', colors(i,:), ...
                        'Tag', sprintf('pkg%uplot', i));
                end
            end

            % Update rectangles
            for i = 1:numBoxes
                if isgraphics(this.BoxHandles(i))
                    this.BoxHandles(i).Position = [x_boxes(i), y_boxes(i), d_boxes(i)/2, d_boxes(i)/2];
                end
            end

            % Create text annotations if not already initialized
            if isempty(this.BoxTextHandles) || any(~isgraphics(this.BoxTextHandles))
                this.BoxTextHandles = gobjects(1, numBoxes);
                for i = 1:numBoxes
                    this.BoxTextHandles(i) = text(ha, ...
                        x_boxes(i) + d_boxes(i)/4, ...
                        y_boxes(i) + d_boxes(i)/2, ...
                        sprintf('%.2f', d_boxes(i)), ...
                        'HorizontalAlignment', 'center', ...
                        'VerticalAlignment', 'bottom', ...
                        'FontSize', 10, ...
                        'Color', 'k', ...
                        'FontWeight', 'bold');
                end
            end

            % Update text annotations
            for i = 1:numBoxes
                if isgraphics(this.BoxTextHandles(i))
                    this.BoxTextHandles(i).Position = [x_boxes(i) + d_boxes(i)/4, y_boxes(i) + d_boxes(i)/2];
                    this.BoxTextHandles(i).String = sprintf('%.2f', d_boxes(i));
                end
            end

            
            % drawnow limitrate
            drawnow();
            if env.SaveGIF
                frame = getframe(this.Figure);
                im = frame2im(frame);
                im = imresize(im, 0.5);
                [imind, cm] = rgb2ind(im, 64);
                
                
               
                frameDelay = 0.02;

                if env.GIFFrameCount == 1
                    imwrite(imind, cm, env.GIFFile, 'gif', 'Loopcount', inf, 'DelayTime', frameDelay);
                elseif mod(env.GIFFrameCount,3)==0
                    imwrite(imind, cm, env.GIFFile, 'gif', 'WriteMode', 'append', 'DelayTime', frameDelay);
                end

                env.GIFFrameCount = env.GIFFrameCount + 1;
            end
        end
    end
end