function []= export_to_file(reflectivities, file_name, number_of_frequencies, file_type) 
% reflectivities_with_noise: Reflectivities to add to the file
% file_name: Beginning of the file name
% number_of_frequencies: Total number of frequencies
% thickness: Current thickness
% file_type: Either txt or csv


file_name = strcat(file_name,".",file_type);

% Save data into files
if file_type == "txt"
    file=fopen(file_name,'w');
    for i=1: length(reflectivities)
        row = '';
        for frequency=1:number_of_frequencies
            row = row + " " + num2str(reflectivities(frequency,i));
        end
        row = row + "\n";
        fprintf(file,row);
    end
    
    fclose(file);
end

%% TODO: Fix csv exporter
if file_type == "csv"
    writetable(cast(reflectivities,"single"), file_name);
end


end