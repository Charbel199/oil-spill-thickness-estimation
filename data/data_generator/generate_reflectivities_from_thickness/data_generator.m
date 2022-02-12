%% This code is to generate reflectivities based on the provided frequencies and thicknesses
%% The code is also able to add normal gaussian noise to the generated reflectivities
% -----------------------------------------------------------------------

clear all

% Parameters
f= [4.3855 6.9759 9.0681]*1e9; % frequency range of EM in GHz  4.151,7.743,7.939 or 11
ks=[0 0 0]; 
loss = exp(-4.*(ks*cosd(0)).^2);
%s=0.3e-2;
%ks= (2*pi./lambda).*s
thickness= [0:1:10]; % in mm
c = 3e8; % speed of light
lambda= c./f; % EM wavelength

temperature_w = 20; % water temperature in degrees
salinity_w = 30; % salinity in parts per thousand psu
[epsr_w, epsi_w] = module4_2(temperature_w,f/1e9,salinity_w);
eps1 = 1; % dielectric permittivity of medium 1, air
eps3 = epsr_w-epsi_w*1i; % dielectric permittivity of medium 3, saline water
eps2= 3.3; % dielectric permittivity of medium 2, oil
n1 = sqrt(eps1);
n2= sqrt(eps2);
n3= sqrt(eps3);

variance_noise= 0.01;
file_name = "fromThicknessLowVariance";

r = zeros(length(lambda),length(thickness));
R = zeros(length(lambda),length(thickness)); % Number of frequencies x Number of thicknesses


for p=1:length(lambda) % variation in frequency
    % reflection coefficient for single interfaces
    r1 = (n1-n2)/(n1+n2); % air-oil interface
    r2(p)= (n2-n3(p))/(n2+n3(p)); % oil-water interface
    r_free(p) = (r1+r2(p))./(1+r1*r2(p)); % refl coef
    %R_free(p)= (abs(r_free(p)))^2*loss(p);

    % Variation in Oil Thickness
    delta2 = 2*pi/lambda(p)*n2*0.001.*(thickness); % for 1 frequency value
    r(p,:) = (r1+r2(p).*exp(-1j*2.*delta2))./(1+r1*r2(p).*exp(-1j*2.*delta2)); % refl coef
    R(p,:)= (abs(r(p,:))).^2*loss(p); % Reflectivity
end

for s=1:length(thickness)
    reflectivities_with_noise = noise(R(:,s),thickness(s), variance_noise);
    export_to_file(reflectivities_with_noise, file_name, size(R(:,s),1), thickness(s));
end
