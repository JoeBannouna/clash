import click
from src.omega_calculator import OmegaCalculator

@click.command()
@click.option("--tel-pos", type=(float, float), required = True, help='Centre of telescope on 2D plane')
@click.option("--r-cone", type = float, required = True, help = 'Radius of cherenkov cone')
@click.option("--r-tel-reach", type = float, required = True, help = 'Radius of outer circle of telescope')
@click.option("--r-tel-core", type = float, required = True, help = 'Radius of inner circle of telescope')
@click.option('--export-path', type=click.Path(), default=None, help='Sets export path for plots')

# # Case 8
# # Test 1
# omega_calculator.update(cone_pos = (1, 1), tel_pos = (2.5, 0.4), r_cone = 1, r_tel_core = 1.3, r_tel_reach = 2)
# omega_calculator.graphCircles()
# omega = omega_calculator.getOmegas()
# print("Omega: ", omega, " degrees.")

# python3 -m scripts.main --tel-pos 1.5 -0.6 --r-cone 1.0 --r-tel-reach 2.0 --r-tel-core 1.3 --export-path scripts/circles_plot.png

def main(tel_pos, r_cone, r_tel_reach, r_tel_core, export_path):
    omega_calc = None

    try:
        omega_calc = OmegaCalculator()
    except Exception as e:
        click.echo(f"Error instantiating class: {e}")
        raise click.Abort()
    
    try:
        omega_calc.update(cone_pos = (0.0, 0.0), tel_pos=tel_pos, r_cone=r_cone, r_tel_reach=r_tel_reach, r_tel_core=r_tel_core)
    except Exception as e:
        click.echo(f"Error updating information in OmegaCalculator: {e}")
        raise click.Abort()
    
    try:
        omega = omega_calc.getOmegas()
        print(omega)
    except Exception as e:
        click.echo(f"Error calculating omega: {e}")
        raise click.Abort()
    
    try:
        if export_path:
            omega_calc.graphCircles(export_path = export_path, custom_title = f"Shower Cone and Telescope Projection (δω = {omega:.2f})")
        else:
            omega_calc.graphCircles()
    except Exception as e:
        click.echo(f"Error plotting graph")
        raise click.Abort()
    
if __name__ == '__main__':
    main()