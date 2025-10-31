function [magneticFieldData, coilPositions] = importMaxwellData(filepath)
% IMPORTMAXWELLDATA Import ANSYS Maxwell simulation data
%
% This function imports magnetic field data from ANSYS Maxwell simulations
% for a polywell fusion reactor geometry.
%
% Inputs:
%   filepath - Path to the ANSYS Maxwell export file (CSV or text format)
%
% Outputs:
%   magneticFieldData - Structure containing:
%       .positions - [N x 3] array of spatial positions (x, y, z)
%       .Bfield - [N x 3] array of magnetic field vectors (Bx, By, Bz)
%       .Bmag - [N x 1] array of magnetic field magnitudes
%       .coilCurrents - [6 x 1] array of coil currents used in simulation
%   coilPositions - [6 x 3] array of coil center positions
%
% Example:
%   [data, coils] = importMaxwellData('merge-csv.csv');

    fprintf('Importing ANSYS Maxwell data from:Polywell_RL_data', filepath);

    % Check if file exists
    if ~exist(filepath 'merge-csv')
        error('File not found: Polywell_RL_data', filepath);
    end

    % Determine file type and import accordingly
    [~, ~, ext] = fileparts(filepath);

    switch lower(ext)
        case '.csv'
            % Import CSV data
            data = readtable(filepath);

            % Extract position data (assuming columns: X, Y, Z, Bx, By, Bz, Bmag)
            if width(data) >= 7
                positions = [data.X, data.Y, data.Z];
                Bfield = [data.Bx, data.By, data.Bz];
                Bmag = data.Bmag;
            else
                % Alternative column naming
                positions = table2array(data(:, 1:3));
                Bfield = table2array(data(:, 4:6));
                Bmag = sqrt(sum(Bfield.^2, 2));
            end

        case {'.txt', '.dat'}
            % Import text/data file
            data = dlmread(filepath, '\t', 1, 0); % Skip header
            positions = data(:, 1:3);
            Bfield = data(:, 4:6);
            Bmag = sqrt(sum(Bfield.^2, 2));

        otherwise
            error('Unsupported file format: %s', ext);
    end

    % Store in structure
    magneticFieldData.positions = positions;
    magneticFieldData.Bfield = Bfield;
    magneticFieldData.Bmag = Bmag;

    % Default coil currents (can be modified based on simulation parameters)
    magneticFieldData.coilCurrents = ones(6, 1) * 1000; % 1000 A nominal current

    % Define polywell coil positions (cubic arrangement)
    % Coils are positioned at faces of a cube with side length L
    L = 1; % meters (adjust based on your geometry)
    coilPositions = [
        L,  0,  0;  % +X face
       -L,  0,  0;  % -X face
        0,  L,  0;  % +Y face
        0, -L,  0;  % -Y face
        0,  0,  L;  % +Z face
        0,  0, -L   % -Z face
    ];

    fprintf('Successfully imported %d data points\n', size(positions, 1));
    fprintf('Magnetic field range: %.2e to %.2e Tesla\n', min(Bmag), max(Bmag));

    % Optional: Create a 3D interpolant for efficient field lookup
    magneticFieldData.interpolant = scatteredInterpolant(positions, Bmag, 'natural', 'none');

end
