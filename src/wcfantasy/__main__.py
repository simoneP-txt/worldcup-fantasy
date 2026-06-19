import argparse
import sys

from . import advise, calibrate, fifa_client, live

# Windows consoles often default to cp1252; player names are UTF-8.
sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(prog="wcfantasy")
    sub = parser.add_subparsers(dest="command", required=True)

    advise_p = sub.add_parser("advise", help="full recommendation for the next round")
    advise_p.add_argument(
        "--from-scratch",
        action="store_true",
        help="build the unconstrained ideal 15 (ignoring the saved team and "
        "transfer limits) and show how far your current squad is from it; "
        "reference only, not saved",
    )
    sub.add_parser("fetch", help="refresh FIFA data only")
    sub.add_parser(
        "odds", help="model championship probability (P win) for each team"
    )

    live_p = sub.add_parser(
        "live", help="intra-round advice: captain switches and manual subs"
    )
    live_p.add_argument(
        "--mark-changed",
        action="store_true",
        help="record that a manual change was made in the game this round "
        "(cancels auto-subs and vice insurance)",
    )
    live_p.add_argument(
        "--captain",
        metavar="NAME",
        help="record that you changed the captain to this player of your "
        "squad (implies --mark-changed)",
    )
    live_p.add_argument(
        "--sub-out",
        metavar="NAME",
        help="record a manual substitution: the starter you moved OUT "
        "(use with --sub-in; implies --mark-changed)",
    )
    live_p.add_argument(
        "--sub-in",
        metavar="NAME",
        help="the bench player you moved IN (use with --sub-out)",
    )
    live_p.add_argument(
        "--vice",
        metavar="NAME",
        help="record that you changed the vice-captain to this player "
        "(implies --mark-changed)",
    )

    team = sub.add_parser("team", help="manage local team state")
    team.add_argument("action", choices=["show", "apply", "use-booster"])
    team.add_argument(
        "booster",
        nargs="?",
        help="booster name to mark as used (with the use-booster action)",
    )
    team.add_argument(
        "--force",
        action="store_true",
        help="apply even while an earlier round is still live (overwrites "
        "the lineup `live` reads for that round)",
    )

    cal = sub.add_parser(
        "calibrate", help="refit the model priors against real WC results"
    )
    cal.add_argument(
        "--write",
        action="store_true",
        help="persist the proposal to data/state/calibration.json "
        "(default is a dry run)",
    )

    serve_p = sub.add_parser(
        "serve", help="start the local web app (and open the browser)"
    )
    serve_p.add_argument(
        "--port", type=int, default=None,
        help="port to serve on (default 8765)",
    )
    serve_p.add_argument(
        "--no-browser", action="store_true",
        help="do not open the browser automatically",
    )

    args = parser.parse_args()
    if args.command == "advise":
        advise.run(from_scratch=args.from_scratch)
    elif args.command == "fetch":
        changed = fifa_client.refresh()
        print({k: ("updated" if v else "unchanged") for k, v in changed.items()})
    elif args.command == "odds":
        advise.show_odds()
    elif args.command == "live":
        live.run(
            mark_changed=args.mark_changed,
            captain=args.captain,
            sub_out=args.sub_out,
            sub_in=args.sub_in,
            vice=args.vice,
        )
    elif args.command == "calibrate":
        calibrate.run(write=args.write)
    elif args.command == "serve":
        from . import server

        server.serve(
            port=args.port or server.DEFAULT_PORT,
            open_browser=not args.no_browser,
        )
    elif args.command == "team":
        if args.action == "show":
            advise.show_team()
        elif args.action == "apply":
            advise.apply_last_advice(force=args.force)
        else:  # use-booster
            advise.use_booster(args.booster)


if __name__ == "__main__":
    main()
