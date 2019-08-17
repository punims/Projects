package oop.ex6.Variables;
import oop.ex6.Block.*;
import java.util.regex.Pattern;


public class SBoolean extends Variable {

    final public static String BOOLEAN_VALUE = "[\\s]*[\\d]+\\.[\\d]+|[\\d]+[\\s]*|[\\s]*true[\\s]*|[\\s]*false[\\s]*|" + VARIABLE_NAME;

    /**
     * default constructor
     * @param name
     * @param isFinal
     * @param assignment
     */
    public SBoolean(String name, boolean isFinal, boolean assignment, Block currentBlock) {
        super(name, isFinal, assignment, currentBlock);
    }

    @Override
    boolean checkValue(String value) {
        p = Pattern.compile(BOOLEAN_VALUE);
        m = p.matcher(value);
        return m.matches();
    }
}
