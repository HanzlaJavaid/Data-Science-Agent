import sys
import matplotlib.pyplot as plt
from io import StringIO, BytesIO

def execute_code(code,globals):
    # Dedent the code to adjust indentation
    dedented_code = code
    # Redirect stdout
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    # Execute the code
    try:
        print(dedented_code)
        exec(dedented_code,globals)
    except Exception as e:
        # Restore stdout
        sys.stdout = old_stdout
        return {"error": str(e)}

    # Check if a plot was created
    if plt.get_fignums():
        # Save the plot to a BytesIO object
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

        #image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

        output = redirected_output.getvalue()
        # Restore stdout
        sys.stdout = old_stdout

        #return {"output": output, "image_base64": image_base64}
        return {"output": output}

    else:
        output = redirected_output.getvalue()
        # Restore stdout
        sys.stdout = old_stdout

        return {"output": output}


print(execute_code("""\nprint("hello world")""",globals()))