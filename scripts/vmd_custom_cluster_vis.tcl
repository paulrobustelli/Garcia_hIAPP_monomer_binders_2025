# Set display settings
color Display Background 8  ;# Set background color to white
axes location off           ;# Turn off axes
display projection Orthographic  ;# Set display to orthographic mode

# Set color IDs
set color_protein 30
set color_ligand 30

# Apply to all loaded molecules
foreach mol [molinfo list] {

    # Get the total number of frames
    set last_frame [molinfo $mol get numframes]

    # Compute stride to get exactly 10 frames
    if {$last_frame > 10} {
        set stride [expr int($last_frame / 10)]
    } else {
        set stride 1
    }

    # Delete existing representations
    mol delrep 0 $mol 

    # Representation 1: New Cartoon for the protein in frame 0
    mol addrep $mol
    mol modstyle 0 $mol NewCartoon
    mol modcolor 0 $mol ColorID $color_protein
    mol modselect 0 $mol "protein"
    mol drawframes $mol 0 {0}  ;# Show only frame 0

    # Representation 2: Licorice for selected residues in frame 0
    mol addrep $mol
    mol modstyle 1 $mol Licorice
    mol modcolor 1 $mol ColorID $color_protein
    mol modselect 1 $mol "resid 2 7 15 18 23 37 and noh"
    mol drawframes $mol 1 {0}  ;# Show only frame 0

    # Representation 3: Transparent Tube (radius 0.1) for protein, sampled every `stride` frames
    mol addrep $mol
    mol modstyle 2 $mol Tube 0.1
    mol modcolor 2 $mol ColorID $color_protein
    mol modselect 2 $mol "protein"
    mol drawframes $mol 2 "1:$stride:$last_frame"  ;# Show frames at interval `stride`
    mol modmaterial 2 $mol Transparent  ;# Apply transparency

    # Representation 4: Licorice for LIG (noh) in frame 0 
    mol addrep $mol
    mol modstyle 3 $mol Licorice
    mol modcolor 3 $mol Type
    mol modselect 3 $mol "resname LIG and noh"
    mol drawframes $mol 3 {0}  ;# Show only frame 0


    # Representation 5: Transparent Licorice (radius 0.1) for LIG (noh), sampled every `stride` frames
    mol addrep $mol
    mol modstyle 4 $mol Licorice 0.1  ;# Set bond radius to 0.1
    mol modcolor 4 $mol Type
    mol modselect 4 $mol "resname LIG and noh"
    mol drawframes $mol 4 "1:$stride:$last_frame"  ;# Show frames at interval `stride`
    mol modmaterial 4 $mol Transparent  ;# Apply transparency

    # Hide all representations after creating them
    for {set rep 0} {$rep < 5} {incr rep} {
        mol showrep $mol $rep 0  ;# Hide representation
    }

    # ===== ALIGNMENT USING RMSD Trajectory Tool =====
    # Align each molecule's protein to its own frame 0
    set sel [atomselect $mol "protein"]  ;# Select protein
    set ref [atomselect $mol "protein" frame 0]  ;# Select reference frame (frame 0)

    # Loop through all frames and align them
    for {set frame 1} {$frame < $last_frame} {incr frame} {
        $sel frame $frame
        $sel move [measure fit $sel $ref]  ;# Apply RMSD fit transformation
    }

    $sel delete
    $ref delete
}

puts "Custom representations applied, all representations hidden, and proteins aligned to frame 0."
