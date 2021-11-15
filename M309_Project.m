%% MECH 309 - Project: Murman-Cole Scheme for the Transonic Small Disturbance

% By:
% Ahmed Zedan 260662619
% Pierrick Hamard 260619897
% Chris Jing 260634735
% Khaled Al Masaid 260623070

%%%%%%%%  To run each question individually, comment out the other questions  %%%%%%%%%

clc
clear all

%% Question 1

%Section is to test run, check, and build plots for other questions, comment out in final run

% dx = 0.1;
% M_inf = 0.75;
% tol = 1e-04;  %Convergence limit
% max_bnd = 50; %Max limit in both x and y dir (square computational domain, i.e. x,y on [0, max_bnd]^2)
% dy = dx;
% 
% [phi, error, c_p, P, iteration] = TSD_Solve(dx, M_inf, tol, max_bnd);
% 
% X = 0:dx:max_bnd;
% Y = 0:dy:max_bnd;
% 
% %Phi plot over computational domain
% figure(1)
% surface(X, Y, phi)
% title('Velocity Potential (phi) Plot')
% ylabel('y')
% xlabel('x')
% 
% %Setup vectors for Convergence plot and create it
% I = 1:iteration;
% error = error(1:iteration);
% 
% figure(2)
% loglog(I,error)
% title('Convergence Plot')
% ylabel('Error')
% xlabel('Iterations')
% xlim([2 iteration]) %Iteration no. 1 is a manually-set value, ignore it and start from 2 instead
% 
%C_p Contour plot - not required in questions but pretty to look at
% figure(3)
% contour(X, Y, c_p, 15)
% hold on
% contour(X, -Y, c_p, 15); %Mirror results in y
% hold off
% title('Cp Contour Plot')
% ylabel('y')
% xlabel('x')
% xlim([19 22])
% ylim([-2 2])
% 
% %C_p plot over airfoil surface
% figure(4)
% plot(X, c_p(1,:))
% axis ij %Flip y axis so -ve on top
% title('Cp Over AF Surface')
% ylabel('y')
% xlabel('x')
% xlim([20 21])
% ylim([-1 1])


%% Question 2

dx = 0.1;
tol = 1e-04;  %Convergence limit
max_bnd = 50; %Max limit in both x and y dir (square computational domain, i.e. x,y on [0, max_bnd]^2)
dy = dx;

X = 0:dx:max_bnd;
Y = 0:dy:max_bnd;

x_conv = @(i) (dx*(i-1)); %Converts index no. to coordinate no.
i_conv = @(x) x/dx +1; %Converts coordinate no. to index.

x_min = 20;
x_max = 21;
y_min = 0;
y_max = 1;

fig_counter = 3;

for M_inf = 0.75:0.02:0.85
    [phi, error, c_p, P, iteration] = TSD_Solve(dx, M_inf, tol, max_bnd);
    
    %L_inf Norm Convergence plots
    I = 1:iteration;
    error = error(1:iteration);
    figure(1)
    loglog(I,error, 'DisplayName', ['M_{inf}=' num2str(M_inf)])
    title('L_{inf} Norm Convergence Plot')
    ylabel('Log(Error)')
    xlabel('Iteration Number')
    xlim([2 iteration]) %Iteration no. 1 is a manually-set value, ignore it and start from 2 instead
    hold on
        
    %Surface Pressure Coefficient plots as a function of x:
    figure(2)
    plot(X, c_p(1,:), 'DisplayName', ['M_{inf}=' num2str(M_inf)]) %Plot results for current M_inf and name it accordingly
    axis ij %Flip y axis so -ve on top
    title('C_p Over Airfoil Surface')
    ylabel('C_p - Pressure Coefficient')
    xlabel('x - Horizontal Position [m]')
    xlim([20 21])
    hold on
        
    %Pressure contour plot
    figure(fig_counter)
    contour(x_min:dx:x_max, y_min:dy:y_max, P(i_conv(y_min):i_conv(y_max),i_conv(x_min):i_conv(x_max)), 15)
    %hold on
    %contour(x_min:dx:x_max, -(y_min:dy:y_max), P(i_conv(y_min):i_conv(y_max),i_conv(x_min):i_conv(x_max))); %Mirror results in y
    %hold off
    title(['P/P_{inf} Contour Plot for M_{inf} = ' num2str(M_inf)])
    ylabel('y - Vertical Position [m]')
    xlabel('x - Horizontal Position [m]')
    
    fig_counter = fig_counter + 1;
end

figure(1)
legend

figure(2)
legend


%% Question 3

M_inf = 0.85;
tol = 1e-04;  %Convergence limit
max_bnd = 50; %Max limit in both x and y dir (square computational domain, i.e. x,y on [0, max_bnd]^2)

for dx = [0.1 0.05 0.025]
    [phi, error, c_p, P, iteration] = TSD_Solve(dx, M_inf, tol, max_bnd);
    
    dy = dx;
    X = 0:dx:max_bnd;
    Y = 0:dy:max_bnd;
    
    figure(1)
    plot(X, c_p(1,:), 'DisplayName', ['dx=' num2str(dx)]) %Plot results for current grid spacing and name it accordingly
    axis ij %Flip y axis so -ve on top
    title(['C_p Over Airfoil Surface for Range of Grid Sizes @ M_{inf}=' num2str(M_inf)])
    ylabel('C_p - Pressure Coefficient')
    xlabel('x - Horizontal position [m]')
    xlim([20 21])
    hold on
end

legend
 

%% Question 4

dx = 0.025;
tol = 1e-04;  %Convergence limit
max_bnd = 50; %Max limit in both x and y dir (square computational domain, i.e. x,y on [0, max_bnd]^2)
X = 0:dx:max_bnd;

%C_p plot over airfoil surface for range on M_inf
figure(5)

for M_inf = 0.75:0.02:0.85
    [phi, error, c_p, P, iteration] = TSD_Solve(dx, M_inf, tol, max_bnd);
    
    plot(X, c_p(1,:), 'DisplayName', ['M_{inf}=' num2str(M_inf)]) %Plot results for current M_inf and name it accordingly
    
    hold on
    
    if M_inf == 0.75 %Setup plot only once after initial solution run
        axis ij %Flip y axis so -ve on top
        title('Cp Over AF Surface for M_{inf} [0.75 to 0.85]')
        ylabel('C_p - Pressure Coeff.')
        xlabel('x - Horizontal Position [m]')
        xlim([20 21])
        ylim([-1 1])
    end
end

legend


%% The TSD Solver Function: 

function [phi, error, c_p, P, iteration] = TSD_Solve(dx, M_inf, tol, max_bnd)

% Constants:
gamma = 1.4;
R = 287.058; %J/(kg*K)
T_inf = 293; %K
P_inf = 100; %kPa
t_ratio = 0.08;
dy = dx; %m?

% Initializing grid (square grid):
n = max_bnd/dx + 1; % grid size for specified dx (x,y [0,50]^2 -> 1:501 for index interval of 0.1)
%points = n^2; %total number of points in grid
phi = zeros(n,n); %grid
A = zeros(n,n); % variable to determine subsonic or supersonic flow

% Initialize various const. matricies and vectors
MU = zeros(n,n);
phi_new = zeros(n,n);
a = zeros(n,n);
b = zeros(n,n);
c = zeros(n,n);
d = zeros(n,n);
e = zeros(n,n);
g = zeros(n,n);
iteration = 1;
error = zeros(1,50000);
error(1) = 1e+05;

%Airfoil Slope:
AF_slope = @(x) (t_ratio*(-4*x + 82)); 
%AF = @(x) (t_ratio*(-2*x.^2+82*x-840));

%Conversion function from index no. to x value
x_conv = @(i) (dx*(i-1));

%U_inf as func of M_inf
U_inf = M_inf*sqrt(gamma*R*T_inf);

%Setting Boundary Conditions (top, left, right edges):
phi(:,1) = 0;
phi(n,:) = 0;
phi(:,n) = 0;

tic

while abs(error(iteration)) > tol
    
    iteration = iteration + 1;
    
    %Set boundary conditions (along bottom edge)   
    for i = 2:n-1
        if x_conv(i) >= 20 && x_conv(i) <= 21
            phi(1,i) = phi(2,i) - U_inf*dy*AF_slope(x_conv(i));
        else
            phi(1,i) = phi(2,i);       
        end
    end

    %Computing Ai,j values, MU based on switch, and all function coefficients:
    for j = 2 : n-1
        for i = 2 : n-1
            A(j,i) = (1-M_inf(1).^2) - (gamma+1)*((M_inf(1).^2)/U_inf)*((phi(j,i+1)-phi(j,i-1))/(2*dx));
            
            if A(j,i) > 0
                MU(j,i) = 0;
            else
                MU(j,i) = 1;
            end  
            
            a(j,i) = ((MU(j,i-1)*A(j,i-1))/(dx.^2)) - ((2*(1-MU(j,i))*A(j,i))/(dx.^2)) - (2/(dy.^2));
            b(j,i) = (1/(dy.^2));
            c(j,i) = (1/(dy.^2));
            d(j,i) = (((1-MU(j,i))*A(j,i))/(dx.^2)) - ((2*MU(j,i-1)*A(j,i-1))/(dx.^2));
            e(j,i) = ((1-MU(j,i))*A(j,i))/(dx.^2);
            g(j,i) = (MU(j,i-1)*A(j,i-1))/(dx.^2); 
        end
    end

    %Calculate internal phi values using stencil, keep track of max element error between iterations,
    %then update current phi values:
    for j = 2:n-1
        for i = 3:n-1
            phi_new(j,i)= (-c(j,i)*phi(j-1,i) - g(j,i)*phi(j,i-2) - d(j,i)*phi(j,i-1) - e(j,i)*phi(j,i+1) - b(j,i)*phi(j+1,i))/(a(j,i));
        
            error(iteration) = max(error(iteration), abs(phi_new(j,i)-phi(j,i)));
            
            phi(j,i)=phi_new(j,i);
        end
    end
               
%     Debug stuff - Stop iterations if solution diverging
%     if error(iteration) > error(iteration-1)
%         fprintf("Solution diverging: iter. " + iteration + "\n")
%         break;
%     end
   
end

toc

% CP plot: c_p calculated along bottom using central finite difference in x
c_p = zeros(n);
for j = 1:n
    for i = 2:n-1
        c_p(j,i) = -(phi(j,i+1) - phi(j,i-1)) / (U_inf * dx);
    end
end

% Calculate P/P_inf from converged Phi Values:
P = zeros(n);
for j = 1:n
    for i = 2:n-1
        u = (phi(j,i+1)- phi(j,i-1))/(2*dx);
        
        if j == 1
            v = (phi(j+1,i)- phi(j,i))/(dy);
        else
            if j == n
                v = (phi(j,i)- phi(j-1,i))/(dy);
            else    
                v = (phi(j+1,i)- phi(j-1,i))/(2*dy);
            end
        end    
        
        P(j,i) = ((1 + ((gamma-1)/2) * (M_inf.^2) * (1 - ((u^2 + v^2)/(U_inf^2)))) .^ (gamma /(gamma-1)));
    end
end

end
