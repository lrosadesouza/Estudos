package A2;
import java.io.*;
import java.util.*;

public class RLA2{
    static int[][] track;

    public static int[][] generateTrack(int rows, int cols){
        int[][] track = new int[rows][cols];
        try{
            File file = new File("/home/lucas/Desktop/TUFTS/Fall 22/CS 138 - RL/CodingAssignments/A2/Racetrack1");
            System.out.println(file.canRead());
            BufferedReader reader = new BufferedReader(new FileReader(file)); 
            String line;
            String newLine = new String("");
            ArrayList<String> newLines = new ArrayList<>();
            int i = 0; 
            while((line = reader.readLine()) != null){
                for(int j = 0; j < cols; j++)
                {
                    track[i][j] = Integer.parseInt(line.charAt(j) + "");
                    newLine += track[i][j] + (j == (cols - 1) ? "" : ",");
                }
                i++;
                newLines.add(newLine);
                newLine = "";
            }
            reader.close();

            BufferedWriter writer = new BufferedWriter(new FileWriter(file));

            for(String line2 : newLines)
            {
                writer.write(line2);
                writer.newLine();
            }

            writer.close();
        }
        catch(Exception e){} 

        return track;
    }

    public static void main(String[] args)
    {
        track = generateTrack(32, 25);

        System.out.println(Arrays.deepToString(track));
        
    }
}