package oop.ex6.main;
import oop.ex6.Block.Global;
import oop.ex6.Exceptions.BlockException;
import oop.ex6.Exceptions.IllegalLineException;
import oop.ex6.Exceptions.VariableException;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;


public class Sjavac {

    private static final String ERROR = "1", CORRECT = "0", IOEXCEPTION = "2";


    static final int FILE_LOCATION = 0;
    private static ArrayList<String> lines;


    /**
     * the main function which runs the compiler.
     * @param args
     */
    public static void main(String[] args) {
        try {
            File f = new File(args[FILE_LOCATION]);
            FirstScan firstScan = new FirstScan(f);
            lines = firstScan.firstLook(f);
            SecondScan secondScan = new SecondScan(lines);
            secondScan.secondLook(lines);
            System.out.println(CORRECT);
        } catch (IllegalLineException | BlockException | VariableException | NullPointerException e) {
            System.out.println(ERROR);
        } catch (IOException e) {
            System.out.println(IOEXCEPTION);
        } finally {
            Global.reset();
            lines = null;
        }
    }
}