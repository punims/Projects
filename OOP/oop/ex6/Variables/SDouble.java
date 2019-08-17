package oop.ex6.Variables;
import oop.ex6.Block.*;
import java.util.regex.Pattern;


public class SDouble extends Variable{

    final public static String DOUBLE_VALUE = "[\\s]*[\\d]+\\.[\\d]+|[\\d]+[\\s]*|" + VARIABLE_NAME;

    /**
     * default constructor
     * @param name
     * @param isFinal
     * @param assignment
     */
    public SDouble(String name, boolean isFinal, boolean assignment, Block currentBlock) {
        super(name, isFinal, assignment, currentBlock);
    }

    @Override
    boolean checkValue(String value) {
        p = Pattern.compile(DOUBLE_VALUE);
        m = p.matcher(value);
        return m.matches();
    }
}
